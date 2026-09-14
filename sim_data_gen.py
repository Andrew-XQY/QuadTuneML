"""Generate the seven-input alpha/momentum-spread pilot with isolated MAD-X jobs.

Run ``python sim_data_gen.py --help``. All generated files stay in repo results/.
The external lattice is read and snapshotted; its original driver is never run.
Native MAD-X TRACK supports this lattice's MATRIX elements and thick quadrupoles.
Neither backend configures space charge. PTC is selectable, but rejects MATRIX.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_KEYS = ["mean_x", "mean_y", "sigma_x", "sigma_y", "emittance_x",
               "emittance_y", "transmission"]
_WORKER = {}
PHYSICS_FUNCTIONS = ["build_quad_sobol_queue", "delta_to_pt", "pt_to_delta", "rms_emittance",
    "generate_particles", "write_inrays", "parse_trackone_to_df", "calculate_beam_parameters",
    "lattice_files", "snapshot_sources", "build_runner", "_initialize_worker", "run_single"]
# The verified 5,000-row pilot predates the separate physics fingerprint.
LEGACY_GENERATOR_SHA256 = "abf037130fab2f11b7832b9db509617ee6204c199a47e4dc4db01c2f916abb5a"
LEGACY_PHYSICS_SHA256 = "480d5e1e891f9f0739a266da13e431f3ed5717708b25c2fcf5dcfd00fc375b7d"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_implementation_sha256():
    """Fingerprint simulation/sampling functions independently of batch I/O."""
    tree = ast.parse(Path(__file__).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    payload = "\n".join(ast.dump(functions[name], include_attributes=False) for name in PHYSICS_FUNCTIONS)
    return hashlib.sha256(payload.encode()).hexdigest()


def diagnostic_columns(settings):
    return list(settings) + ["status", "seconds", "realized_sigma_delta", "realized_mean_delta",
        "realized_sigma_pt", "beta0", "realized_alfx", "realized_gemx", "realized_alfy",
        "realized_gemy", "p0_gev_c", "tracked_initial_sigma_delta", "final_particles",
        "final_centered_emittance_x", "final_centered_emittance_y", "actual_segment_names",
        "lost_particles", "nonfinite_lost_particles", "finite_lost_particles", "loss_locations",
        "nonfinite_rows", "nonfinite_segments", "madx_version", "error"]


def validate_extension(config, previous_dir, queue):
    """Validate a complete deterministic prefix before reusing any records.

    Old CSVs did not have contemporaneous checksums. Their schema, inputs,
    diagnostics and provenance are checked, then their present hashes are saved.
    """
    from cpymad.madx import Madx
    previous_dir = Path(previous_dir).resolve()
    metadata_path = previous_dir / "metadata.json"
    previous = json.loads(metadata_path.read_text())
    count = previous.get("requested", 0)
    if previous.get("status") != "complete" or previous.get("failed") != 0 or previous.get("completed") != count or not 0 < count < len(queue):
        raise ValueError("Reuse requires a complete, failure-free, shorter simulation prefix")
    if previous.get("sampling", "sobol") != "sobol":
        raise ValueError("Only a deterministic Sobol prefix can be extended")
    old_config = previous["configuration"]
    keys = ("particles", "seed", "backend", "quad_keys", "quad_range", "alpha_fraction",
            "sigma_delta_range", "beam", "observations", "expected_segments",
            "lattice_root", "external_runner", "initial_conditions")
    changed = [key for key in keys if old_config.get(key) != config.get(key)]
    if changed:
        raise ValueError("Simulation configuration changed: " + ", ".join(changed))
    fingerprint = physics_implementation_sha256()
    prior_fingerprint = previous.get("physics_implementation_sha256")
    if prior_fingerprint is None:
        if previous.get("generator_sha256") != LEGACY_GENERATOR_SHA256 or fingerprint != LEGACY_PHYSICS_SHA256:
            raise ValueError("Unrecognized legacy simulator or changed simulation implementation")
    elif prior_fingerprint != fingerprint:
        raise ValueError("Simulation implementation changed")
    expected_sources = [Path(config["lattice_root"]) / name for name in lattice_files()]
    expected_sources += [Path(config[key]) for key in ("external_runner", "initial_conditions")]
    if set(previous["source_sha256"]) != {str(path) for path in expected_sources}:
        raise ValueError("Source-file provenance differs")
    for source in expected_sources:
        if file_sha256(source) != previous["source_sha256"][str(source)]:
            raise ValueError("External simulation source changed: " + str(source))
    template_hash = hashlib.sha256(build_runner(config, queue[0]).encode()).hexdigest()
    if template_hash != previous["runner_template_sha256"] or file_sha256(previous_dir / "runner_template.madx") != template_hash:
        raise ValueError("Generated MAD-X runner differs from previous run")
    for name, version in previous["package_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Runtime package changed: " + name)
    with Madx(stdout=False) as madx:
        if str(madx.version) != previous["madx_version"]:
            raise ValueError("MAD-X version changed")
    data_path = previous_dir / Path(old_config["output"]).name
    diagnostics_path = previous_dir / "diagnostics.csv"
    for key, path in (("data_sha256", data_path), ("diagnostics_sha256", diagnostics_path)):
        if previous.get(key) and file_sha256(path) != previous[key]:
            raise ValueError("Prior CSV checksum no longer matches its metadata: " + str(path))
        if not path.read_bytes().endswith(b"\n"):
            raise ValueError("Prior CSV must end with a complete newline-terminated record")
    data = pd.read_csv(data_path, float_precision="round_trip")
    diagnostics = pd.read_csv(diagnostics_path, float_precision="round_trip", keep_default_na=False)
    if len(data) != count or len(diagnostics) != count:
        raise ValueError("Prior CSV row counts disagree with metadata")
    if list(data.columns) != list(queue[0]) + OUTPUT_KEYS or list(diagnostics.columns) != diagnostic_columns(queue[0]):
        raise ValueError("Prior CSV schema differs")
    expected = pd.DataFrame(queue[:count])
    for label, frame in (("data", data), ("diagnostics", diagnostics)):
        if not np.array_equal(frame[expected.columns].to_numpy(), expected.to_numpy()):
            raise ValueError("Prior {} is not the exact deterministic Sobol prefix".format(label))
    if not (diagnostics.status == "ok").all() or not (diagnostics.nonfinite_rows == 0).all():
        raise ValueError("Prior diagnostics contain failed or nonfinite simulations")
    if not (diagnostics.final_particles + diagnostics.lost_particles == config["particles"]).all():
        raise ValueError("Prior particle-loss accounting is inconsistent")
    if not (diagnostics.finite_lost_particles + diagnostics.nonfinite_lost_particles == diagnostics.lost_particles).all():
        raise ValueError("Prior numerical-loss classifications are inconsistent")
    if not (diagnostics.madx_version == previous["madx_version"]).all():
        raise ValueError("Prior diagnostics contain mixed MAD-X versions")
    for key in OUTPUT_KEYS:
        values = np.asarray([ast.literal_eval(value) for value in data[key]], dtype=float)
        if values.shape != (count, 8) or not np.isfinite(values).all():
            raise ValueError("Invalid prior eight-segment output: " + key)
        if key.startswith("emittance_") and not (values > 0).all():
            raise ValueError("Invalid prior emittance values")
        if key == "transmission":
            if not ((values >= 0) & (values <= 1)).all() or not np.allclose(values[:, -1], diagnostics.final_particles/config["particles"], rtol=0, atol=1e-15):
                raise ValueError("Prior transmission disagrees with surviving particle counts")
    if (previous_dir / "failures.jsonl").read_text().strip():
        raise ValueError("Prior failure log is not empty")
    return dict(directory=str(previous_dir), count=count, data_path=str(data_path),
                diagnostics_path=str(diagnostics_path), metadata_path=str(metadata_path),
                data_sha256=file_sha256(data_path), diagnostics_sha256=file_sha256(diagnostics_path),
                metadata_sha256=file_sha256(metadata_path), prior_generator_sha256=previous["generator_sha256"],
                physics_implementation_sha256=fingerprint, madx_version=previous["madx_version"],
                actual_segment_names=previous["actual_segment_names"],
                validation="Exact Sobol inputs, unchanged simulation functions/config/sources/runtime, complete data and particle-loss accounting")


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if len(config["quad_keys"]) != 4:
        raise ValueError("Exactly four quadrupole inputs are required")
    if len(config["expected_segments"]) != 8:
        raise ValueError("Historical output schema requires exactly eight segments")
    if config.get("backend", "madx") not in ("madx", "ptc"):
        raise ValueError("backend must be madx or ptc; there is no silent fallback")
    beam = config["beam"]
    for key in ("gemx", "gemy", "betax", "betay", "kinetic_energy_gev"):
        if not np.isfinite(beam[key]) or beam[key] <= 0:
            raise ValueError("beam.{} must be finite and positive".format(key))
    if not 0 <= config["alpha_fraction"] < 1:
        raise ValueError("alpha_fraction must lie in [0, 1)")
    low, high = config["sigma_delta_range"]
    if not 0 <= low < high:
        raise ValueError("sigma_delta_range must satisfy 0 <= low < high")
    if len(beam["dispersion_pt"]) != 4:
        raise ValueError("dispersion_pt must contain DX, DPX, DY, DPY")
    return config


def build_quad_sobol_queue(config, n, seed=None):
    """Seven-dimensional scrambled Sobol prefix, stable across worker counts."""
    if n < 1:
        raise ValueError("samples must be positive")
    keys = config["quad_keys"] + ["alfx", "alfy", "sigma_delta"]
    ranges = [config["quad_range"] for _ in config["quad_keys"]]
    for key in ("alfx", "alfy"):
        alpha = config["beam"][key]
        width = abs(alpha) * config["alpha_fraction"]
        ranges.append([alpha - width, alpha + width])
    ranges.append(config["sigma_delta_range"])
    bounds = np.asarray(ranges, dtype=float)
    if np.any(bounds[:, 1] < bounds[:, 0]) or not np.isfinite(bounds).all():
        raise ValueError("Invalid scan bounds")
    sampler = qmc.Sobol(d=len(keys), scramble=True,
                        seed=config["seed"] if seed is None else seed)
    unit = sampler.random_base2(m=int(math.ceil(math.log2(n))))[:n]
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return [dict(zip(keys, row), index=index) for index, row in enumerate(values)]


def delta_to_pt(delta, beta0):
    """Exact PT=(E-E0)/(p0*c), rationalized to avoid low-energy cancellation.

    From (1+delta)^2 = 1 + 2*PT/beta0 + PT^2. See CERN's
    MAD-X canonical variables: https://indico.cern.ch/event/350735/contributions/
    826022/attachments/693330/952005/madX_Nov_6_2014.pdf (slide 3).
    """
    delta = np.asarray(delta, dtype=float)
    if not 0 < beta0 <= 1 or not np.isfinite(delta).all() or np.any(delta <= -1):
        raise ValueError("Require physical beta0 and finite delta > -1")
    change = delta * (2.0 + delta)
    return beta0 * change / (np.sqrt(1.0 + beta0 * beta0 * change) + 1.0)


def pt_to_delta(pt, beta0):
    """Stable inverse used for independent tracking/serialization validation."""
    pt = np.asarray(pt, dtype=float)
    change = 2.0 * pt / beta0 + pt * pt
    return change / (np.sqrt(1.0 + change) + 1.0)


def rms_emittance(x, px, centered=False):
    if centered:
        x, px = x - np.mean(x), px - np.mean(px)
    x2, p2, xp = np.mean(x * x), np.mean(px * px), np.mean(x * px)
    determinant = x2 * p2 - xp * xp
    if determinant < -1e-10 * max(x2 * p2, np.finfo(float).tiny):
        raise ValueError("Nonphysical covariance determinant")
    return float(np.sqrt(max(0.0, determinant)))


def generate_particles(config, settings, beta0):
    """Match pymadx GaussGenerator's transverse covariance in canonical x,px.

    Use a seed per simulation, so parallel scheduling cannot change the beam.
    Transverse geometric emittances retain the old generator's convention.
    Incoming dispersive correlations are added in the MAD-X PT convention.
    """
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], settings["index"]]))
    z = rng.normal(size=(config["particles"], 6))
    rays = np.empty_like(z)
    beam = config["beam"]
    betatron = []
    for axis, offset in (("x", 0), ("y", 2)):
        emit, beta, alpha = beam["gem" + axis], beam["beta" + axis], settings["alf" + axis]
        x = np.sqrt(emit * beta) * z[:, offset]
        px = np.sqrt(emit / beta) * (z[:, offset + 1] - alpha * z[:, offset])
        rays[:, offset], rays[:, offset + 1] = x, px
        betatron.append((x.copy(), px.copy()))
    delta = settings["sigma_delta"] * z[:, 5]
    rays[:, 4] = beam["sigmat"] * z[:, 4]
    rays[:, 5] = delta_to_pt(delta, beta0)
    rays[:, :4] += rays[:, 5, None] * np.asarray(beam["dispersion_pt"])
    stats = {"realized_sigma_delta": float(np.std(delta)),
             "realized_mean_delta": float(np.mean(delta)),
             "realized_sigma_pt": float(np.std(rays[:, 5])), "beta0": float(beta0)}
    for axis, (x, px) in zip(("x", "y"), betatron):
        emit = rms_emittance(x, px, centered=True)
        stats["realized_alf" + axis] = float(-np.mean((x-x.mean())*(px-px.mean())) / emit)
        stats["realized_gem" + axis] = emit
    return rays, stats


def write_inrays(path, rays, backend="ptc"):
    with open(path, "w") as stream:
        for row in rays:
            command = "ptc_start" if backend == "ptc" else "start"
            stream.write(command + ", x={:.17g}, px={:.17g}, y={:.17g}, py={:.17g}, t={:.17g}, pt={:.17g};\n".format(*row))


def parse_trackone_to_df(filepath):
    rows, segments, names, columns = [], [], [], None
    segment = 0
    with open(filepath) as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("*"):
                columns = line[1:].split()
            elif line.startswith("#segment"):
                segment += 1
                names.append(line.split()[-1])
            elif line and not line.startswith(("@", "$", "#")):
                parts = line.split()
                if columns is None or segment == 0 or len(parts) != len(columns):
                    raise ValueError("Malformed TRACKONE data line")
                rows.append([float(x.replace("D", "E")) for x in parts])
                segments.append(segment)
    if not rows:
        raise ValueError("Empty TRACKONE output")
    frame = pd.DataFrame(rows, columns=columns)
    frame["segment"] = segments
    frame.attrs["segment_names"] = names
    return frame


def calculate_beam_parameters(frame, particles=None, expected_segments=None):
    """Preserve historical raw-second-moment canonical emittance output lists."""
    required = ["X", "PX", "Y", "PY", "T", "PT", "S"]
    if not all(key in frame for key in required) or not np.isfinite(frame[required].values).all():
        raise ValueError("Missing or nonfinite tracking coordinates")
    names = frame.attrs.get("segment_names", [])
    if expected_segments and [s.lower() for s in names] != [s.lower() for s in expected_segments]:
        raise ValueError("Unexpected TRACKONE segment names: {}".format(names))
    groups = frame.groupby("segment", sort=True)
    n0 = int(groups.size().iloc[0])
    if particles is not None and n0 != particles:
        raise ValueError("Initial particle count differs from requested count")
    if len(groups) != len(names):
        raise ValueError("An observation segment has no tracked particles")
    records = []
    for segment, group in groups:
        x, px, y, py = [group[key].to_numpy() for key in ("X", "PX", "Y", "PY")]
        records.append(dict(segment=segment, mean_x=float(x.mean()), mean_y=float(y.mean()),
                            sigma_x=float(x.std()), sigma_y=float(y.std()),
                            emittance_x=rms_emittance(x, px), emittance_y=rms_emittance(y, py),
                            transmission=len(group) / n0))
    result = pd.DataFrame(records).set_index("segment")
    if not np.isfinite(result.values).all() or (result[["emittance_x", "emittance_y"]] <= 0).any().any():
        raise ValueError("Nonfinite or nonpositive output emittance")
    return result


def lattice_files():
    return ["deflectors.ele"] + ["{0}/{0}{1}".format(line, suffix)
        for line in ("lne00", "lne01", "lne02") for suffix in (".ele", "_k.str", ".seq")]


def snapshot_sources(config, output_dir):
    """Copy only known static model files. Never execute the external driver."""
    snapshot = output_dir / "source_lattice"
    snapshot.mkdir()
    hashes = {}
    for relative in lattice_files():
        source = Path(config["lattice_root"]) / relative
        text = source.read_text()
        commands = re.sub(r"/\*.*?\*/|!.*?$", "", text, flags=re.S | re.M)
        if re.search(r"(?:^|;)\s*(?:system|call|stop|match|exec)\b", commands, re.I):
            raise ValueError("Unexpected executable control command in {}".format(source))
        target = snapshot / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(str(source), str(target))
        hashes[str(source)] = hashlib.sha256(target.read_bytes()).hexdigest()
    for key in ("external_runner", "initial_conditions"):
        source = Path(config[key])
        target = output_dir / ("original_" + source.name)
        shutil.copy2(str(source), str(target))
        hashes[str(source)] = hashlib.sha256(target.read_bytes()).hexdigest()
    return snapshot, hashes


def build_runner(config, settings):
    """Standalone tracking driver; no matching, shell commands or source writes."""
    beam = config["beam"]
    lines = ["option, -echo;", "option, rbarc=false;", "beam, particle=antiproton;",
             "mass=beam->mass;", "Ekin={:.17g};".format(beam["kinetic_energy_gev"]),
             "gamman=1+Ekin/mass;", "beta=sqrt(1-1/gamman^2);",
             "pcn=sqrt(Ekin*(Ekin+2*mass));",
             "beam, particle=antiproton, pc=pcn, exn=6e-6/6, eyn=4e-6/6;"]
    for relative in lattice_files():
        lines.append('call, file="lattice/{}";'.format(relative))
        if relative == "lne00/lne00.seq":
            lines.append("extract, sequence=lne00, from=lne.start.0000, to=lne.lne00.lne01, newname=lne00to01;")
        elif relative == "lne01/lne01.seq":
            lines.append("extract, sequence=lne01, from=lne.start.0100, to=lne.lne01.lne02, newname=lne01to02;")
    lines += ["lne00lne01lne02: sequence, refer=entry, l=10.6361309+6.2625788+7.653745125;",
              "lne00to01, at=0;", "lne01to02, at=10.6361309;", "lne02, at=10.6361309+6.2625788;",
              "endsequence;", "seqedit, sequence=lne00lne01lne02; flatten; endedit;"]
    lines += ["{}={:.17g};".format(key, settings[key]) for key in config["quad_keys"]]
    lines.append("use, sequence=lne00lne01lne02;")
    if config.get("backend", "madx") == "ptc":
        lines += ["ptc_create_universe;",
                  "ptc_create_layout, model=2, method=6, nst=3, exact=true, time=true;",
                  'call, file="inrays.madx";']
        lines += ["ptc_observe, place={};".format(name) for name in config["observations"]]
        lines += ["ptc_track, icase=5, closed_orbit=false, dump, element_by_element=true,",
                  "maxaper={1,1,1,1,1e9,1e9,1e9}, onetable=true, turns=1, ffile=1;",
                  "ptc_track_end;", "ptc_end;"]
    else:
        lines += ["track, onepass=true, onetable=true, dump=true, aperture=true, recloss=true;",
                  'call, file="inrays.madx";']
        lines += ["observe, place={};".format(name) for name in config["observations"]]
        # Broad numerical guards, not physical apertures. Delta creates large
        # arrival-time offsets at 100 keV; do not cut on the longitudinal T.
        lines += ["run, turns=1, maxaper={1,1,1,1,1e9,1e9}, ffile=1;", "endtrack;"]
    return "\n".join(lines) + "\n"


def _initialize_worker(config, snapshot, output_dir):
    work = Path(output_dir) / ".work" / ("worker-" + str(os.getpid()))
    work.mkdir(parents=True)
    shutil.copytree(str(snapshot), str(work / "lattice"))
    _WORKER.update(config=config, work=work)


def run_single(settings):
    """Each job owns a fresh MAD-X subprocess, in a worker-private directory."""
    from cpymad.madx import Madx
    config, work = _WORKER["config"], _WORKER["work"]
    start = time.monotonic()
    stats = dict(settings)
    try:
        for filename in ("trackone", "trackloss", "trackloss.csv"):
            (work / filename).unlink(missing_ok=True)
        with open(work / "madx.log", "w") as logfile:
            with Madx(stdout=logfile) as madx:
                stats["madx_version"] = str(madx.version)
                with madx.chdir(str(work)):
                    madx.input("beam, particle=antiproton;")
                    mass = float(madx.beam.mass)
                    kinetic = config["beam"]["kinetic_energy_gev"]
                    momentum = math.sqrt(kinetic * (kinetic + 2 * mass))
                    beta0 = momentum / (mass + kinetic)
                    rays, realized = generate_particles(config, settings, beta0)
                    stats.update(realized, p0_gev_c=momentum)
                    write_inrays(work / "inrays.madx", rays, config.get("backend", "madx"))
                    script = build_runner(config, settings)
                    (work / "runner.madx").write_text(script)
                    madx.call("runner.madx")
                    stats["lost_particles"] = 0
                    stats["nonfinite_lost_particles"] = 0
                    stats["finite_lost_particles"] = 0
                    stats["loss_locations"] = "{}"
                    if "trackloss" in madx.table:
                        loss = madx.table.trackloss.dframe()
                        stats["lost_particles"] = len(loss)
                        if len(loss):
                            lost_coordinates = loss[["x", "px", "y", "py", "t", "pt"]].values
                            stats["nonfinite_lost_particles"] = int((~np.isfinite(lost_coordinates).all(axis=1)).sum())
                            stats["finite_lost_particles"] = len(loss) - stats["nonfinite_lost_particles"]
                            location = next((key for key in ("name", "element") if key in loss), None)
                            stats["loss_locations"] = json.dumps(loss[location].value_counts().to_dict()) if location else "unavailable"
                            loss.to_csv(work / "trackloss.csv", index=False)
                    for key in config["quad_keys"]:
                        if not np.isclose(float(madx.globals[key]), settings[key], rtol=1e-12, atol=1e-12):
                            raise ValueError("Sampled quadrupole strength was changed: " + key)
        frame = parse_trackone_to_df(work / "trackone")
        nonfinite = ~np.isfinite(frame[["X", "PX", "Y", "PY", "T", "PT", "S"]].values).all(axis=1)
        stats["nonfinite_rows"] = int(nonfinite.sum())
        stats["nonfinite_segments"] = json.dumps(frame[nonfinite].groupby("segment").size().to_dict())
        stats["actual_segment_names"] = json.dumps(frame.attrs["segment_names"])
        if config.get("backend", "madx") == "madx":
            # Normalize only the verified native end label, not arbitrary names.
            frame.attrs["segment_names"] = ["end" if name.lower() == "lne00lne01lne02$end"
                                              else name for name in frame.attrs["segment_names"]]
        beam = calculate_beam_parameters(frame, config["particles"], config["expected_segments"])
        first = frame[frame.segment == 1]
        actual_delta = pt_to_delta(first.PT.to_numpy(), beta0)
        stats["tracked_initial_sigma_delta"] = float(np.std(actual_delta))
        if not np.isclose(stats["tracked_initial_sigma_delta"], stats["realized_sigma_delta"], rtol=1e-8, atol=1e-14):
            raise ValueError("TRACKONE initial PT differs from generated momentum spread")
        final = frame[frame.segment == len(config["expected_segments"])]
        if config.get("backend", "madx") == "madx" and len(final) + stats["lost_particles"] != config["particles"]:
            raise ValueError("Final survivors plus recorded losses do not equal initial particles")
        for axis in ("x", "y"):
            stats["final_centered_emittance_" + axis] = rms_emittance(
                final[axis.upper()].values, final[("p"+axis).upper()].values, centered=True)
        stats.update(status="ok", seconds=time.monotonic()-start,
                     final_particles=len(final), error="")
        return dict(settings, **beam.to_dict(orient="list")), stats
    except Exception as error:
        failure_dir = work.parent.parent / "failed_cases" / str(settings["index"])
        failure_dir.mkdir(parents=True, exist_ok=True)
        for name in ("runner.madx", "inrays.madx", "madx.log", "trackloss.csv"):
            if (work / name).exists():
                shutil.copy2(str(work / name), str(failure_dir / name))
        stats.update(status="failed", seconds=time.monotonic()-start,
                     error="{}: {}".format(type(error).__name__, error))
        return None, stats


def run_batch(config, repo_root=REPO_ROOT, extend_from=None, queue=None, metadata_extra=None):
    custom_queue = queue is not None
    queue = build_quad_sobol_queue(config, config["samples"]) if queue is None else queue
    if not queue:
        raise ValueError("Simulation queue is empty")
    if custom_queue and extend_from:
        raise ValueError("Custom repeated-simulation queues cannot extend a Sobol run")
    reuse = validate_extension(config, extend_from, queue) if extend_from else None
    output = Path(config["output"])
    output = (Path(repo_root) / output).resolve() if not output.is_absolute() else output.resolve()
    results_root = (Path(repo_root) / "results").resolve()
    if results_root not in output.parents:
        raise ValueError("Output must be inside this repository's results/ directory")
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in (output.name, "metadata.json", "source_lattice")):
        raise FileExistsError("Use a fresh run directory to preserve previous results")
    if config["particles"] < 3 or config["workers"] < 1:
        raise ValueError("Require particles >= 3 and workers >= 1")
    snapshot, hashes = snapshot_sources(config, output_dir)
    (output_dir / "runner_template.madx").write_text(build_runner(config, queue[0]))
    (output_dir / "simulation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = dict(status="running", configuration=config, source_sha256=hashes,
                    physics_implementation_sha256=physics_implementation_sha256(),
                    sampling="custom" if custom_queue else "sobol",
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    runner_template_sha256=hashlib.sha256((output_dir / "runner_template.madx").read_bytes()).hexdigest(),
                    python=sys.version, executable=sys.executable,
                    package_versions={name: importlib.metadata.version(name) for name in
                                      ("numpy", "pandas", "scipy", "PyYAML", "cpymad")},
                    input_columns=config["quad_keys"]+["alfx", "alfy", "sigma_delta"],
                    output_columns=OUTPUT_KEYS, expected_segments=config["expected_segments"],
                    backend=config.get("backend", "madx"),
                    backend_reason="Native TRACK preserves original MATRIX maps and thick quadrupoles; PTC rejects MATRIX. No automatic fallback.",
                    numerical_guards=[1, 1, 1, 1, 1e9, 1e9], hardware_aperture_model=False,
                    aperture_checks_enabled=True,
                    loss_interpretation="Numerical-domain or broad-guard losses, not measured hardware aperture losses. Emittance targets describe surviving particles; tail loss can reduce them.",
                    space_charge=False, sigma_delta_definition="RMS (p-p0)/p0",
                    pt_definition="(E-E0)/(p0*c); native canonical PT, or PTC time=true",
                    transverse_convention="canonical x,px; historical uncentered RMS emittance targets",
                    dispersion_convention="entrance coordinates += [DX,DPX,DY,DPY]_PT * PT",
                    note="Provisional beam/ranges; independent measured-beam calibration not performed.",
                    completed=0, failed=0, requested=len(queue), started_unix=time.time())
    if set(metadata_extra or {}).intersection(metadata):
        raise ValueError("Custom study metadata must not override simulation provenance")
    metadata.update(metadata_extra or {})
    if reuse:
        metadata.update(reuse= reuse, completed=reuse["count"], reused=reuse["count"],
                        generated_this_run=0, actual_segment_names=reuse["actual_segment_names"],
                        madx_version=reuse["madx_version"])
        shutil.copy2(reuse["data_path"], str(output))
        shutil.copy2(reuse["diagnostics_path"], str(output_dir / "diagnostics.csv"))
        shutil.copy2(reuse["metadata_path"], str(output_dir / "reuse_source_metadata.json"))
        pending = queue[reuse["count"]:]
    else:
        metadata.update(reused=0, generated_this_run=0)
        pending = queue
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    started = time.monotonic()
    diag_keys = diagnostic_columns(queue[0])
    mode = "a" if reuse else "w"
    with open(output, mode, newline="") as data_stream, open(output_dir / "diagnostics.csv", mode, newline="") as diag_stream, open(output_dir / "failures.jsonl", "w") as failures:
        data_writer = csv.DictWriter(data_stream, fieldnames=list(queue[0])+OUTPUT_KEYS)
        diag_writer = csv.DictWriter(diag_stream, fieldnames=diag_keys)
        if not reuse:
            data_writer.writeheader()
            diag_writer.writeheader()
        context = mp.get_context("spawn")
        with context.Pool(config["workers"], initializer=_initialize_worker,
                          initargs=(config, str(snapshot), str(output_dir))) as pool:
            for row, stats in pool.imap(run_single, pending, chunksize=1):
                diag_writer.writerow(stats)
                if row is None:
                    metadata["failed"] += 1
                    failures.write(json.dumps(stats) + "\n")
                    failures.flush()
                else:
                    data_writer.writerow(row)
                    metadata["completed"] += 1
                    metadata["generated_this_run"] += 1
                    if "actual_segment_names" not in metadata:
                        metadata["actual_segment_names"] = json.loads(stats["actual_segment_names"])
                        metadata["madx_version"] = stats["madx_version"]
                done = metadata["completed"] + metadata["failed"]
                if done % 10 == 0 or done == len(queue):
                    data_stream.flush()
                    diag_stream.flush()
                    metadata["elapsed_seconds"] = time.monotonic()-started
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    print("{}/{} completed; {} failed; {:.1f}s".format(
                        done, len(queue), metadata["failed"], metadata["elapsed_seconds"]), flush=True)
    metadata.update(status="complete" if not metadata["failed"] else "complete_with_failures",
                    elapsed_seconds=time.monotonic()-started, finished_unix=time.time())
    metadata["data_sha256"] = file_sha256(output)
    metadata["diagnostics_sha256"] = file_sha256(output_dir / "diagnostics.csv")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    for key in ("samples", "particles", "seed", "workers"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--output", help="CSV path inside this repository's results/")
    parser.add_argument("--backend", choices=("madx", "ptc"))
    parser.add_argument("--extend-from", type=Path,
                        help="Reuse a verified complete Sobol prefix from this prior run directory")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("samples", "particles", "seed", "workers", "output", "backend"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    metadata = run_batch(config, extend_from=args.extend_from)
    return 1 if metadata["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
