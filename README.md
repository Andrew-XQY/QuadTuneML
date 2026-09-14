# ELENA Extraction Beamline ML Toolkit

MAD-X simulation and a neural-network surrogate for the ELENA LNE02 transfer line to AEgIS. The current model predicts final horizontal and vertical emittance from four quadrupole strengths and three incoming-beam scalars: `alfx`, `alfy`, and `sigma_delta = RMS((p-p0)/p0)`.

Completed 12,000-setting experiment: `results/experiment_20260908_12000/README.md`. Data, model, repeatability study, two paper figures and validation records are saved there; the predeclared design is in `protocol.md`. The previous 5,000-setting pilot is retained in `results/pilot_20260907/` (final model: `training_reshuffled/`).

## Current workflow

| File | Role |
| --- | --- |
| `sim_data_gen.py`, `simulation_config.yaml` | Generate deterministic seven-dimensional Sobol settings, sample particles, run isolated MAD-X jobs, and save eight-observation beam outputs. |
| `validate_simulation.py`, `validate_experiment.py`, `tests/` | Physics checks and completed-data audits of provenance, numeric outputs, sampling and particle accounting. |
| `train.py`, `train_config.yaml` | Standalone notebook MLP; save split, fitted transforms, model, predictions, metrics and optional sample-size sweep. |
| `plot_paper.py`, `plot_config.yaml` | Independent entry point for the two paper figures, with per-figure font scaling. |
| `simulation_variability.py` | Repeat fixed settings with independent particle draws; calculate the observed simulation band. |
| `learning_curve_diagnostics.py` | Validation-only Kneedle diagnostics and stability checks for the dashed marker. |
| `analyze.py`, `APS_PLOT_STYLE.py` | Legacy extended analysis and shared scientific styling. |
| `train.ipynb`, `config.yaml`, `utils.py` | Original four-input notebook experiments, retained for reference. |
| `context.md` | One-minute handoff for the next agent. |
| `results/experiment_20260908_12000/` | Current data, training runs, variability study, two paper figures and report. Results are local and Git-ignored. |

## Setup and run

Use a native Python 3.12 environment. The local `.venv` was tested with TensorFlow 2.20 and cpymad 1.19 / MAD-X 5.09.03. On Apple Silicon, the old Intel Conda TensorFlow installation cannot run because it requires AVX. Dependency versions for this execution are in `requirements-lock.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set the source lattice and initial-condition paths in `simulation_config.yaml` for another machine. The supplied paths refer to the locally available ELENA model. Each run copies the necessary source files into its output directory and records their hashes. The external model and its driver are never modified; the original driver is copied for provenance but not executed.

From the repository root, use a fresh output directory for every new run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python validate_simulation.py --output results/validation/new_check --particles 6000
.venv/bin/python validate_simulation.py --beam-input-effects \
  --output results/validation/new_beam_check --particles 6000

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sim_data_gen.py \
  --samples 12000 --particles 6000 --workers 8 \
  --output results/new_experiment/data.csv

.venv/bin/python train.py --data results/new_experiment/data.csv \
  --output-dir results/new_experiment/training --threads 2 --sweep-workers 3 \
  --sample-efficiency 500,1000,2000,3000,4000,5000,6000,7500,9000,9600 \
  --repeat-seeds 42,43,44 --split-seed 42

.venv/bin/python simulation_variability.py --reference-run results/new_experiment \
  --anchors 64 --replicates 24 --workers 2 \
  --output results/new_experiment/variability/data.csv \
  --preprocessing results/new_experiment/training/preprocessing.json

.venv/bin/python learning_curve_diagnostics.py \
  --sweep results/new_experiment/training/sample_efficiency.csv \
  --output results/new_experiment/training/knee_summary.json
```

The September 8 run used `--extend-from results/pilot_20260907` when generating its 12,000 settings: the verified first 5,000 rows are retained byte for byte and 7,000 are newly simulated. Extension checks physics, source hashes, configuration, runtime and the exact Sobol prefix before reusing data. Omit this argument for an entirely fresh batch.

The training defaults allow **500 epochs**, 100 warm-up epochs and 50-epoch validation patience. The best checkpoint is retained across **all** epochs. Explicit seeded epoch-wise reshuffling changes batch order each epoch while reproducing a fresh run. Sample-efficiency fits use three seeds, fixed nested training subsets and validation-only scores; separate spawned processes are reproducible and write isolated outputs. Curves use a common full-training log-target scale. The main seed-42 model alone evaluates the held-out test set.

Training and figure generation can be rerun independently of simulation. Training refuses to overwrite an existing model. Simulation exits with a nonzero status if any setting fails; inspect `metadata.json`, `diagnostics.csv`, `failures.jsonl`, and `failed_cases/` before using an incomplete batch.

## Regenerate the two paper figures

**Plot entry point: `plot_paper.py`. Font settings: `plot_config.yaml`.** Plotting loads saved predictions, learning-curve scores and repeat-study summaries; it never trains or simulates.

```bash
.venv/bin/python plot_paper.py \
  --training-dir results/experiment_20260908_12000/training \
  --output-dir results/experiment_20260908_12000/figures \
  --figure both \
  --variability-summary results/experiment_20260908_12000/variability/variability_summary.json \
  --knee-summary results/experiment_20260908_12000/training/knee_summary.json
```

Use `--figure parity` or `--figure efficiency` to regenerate one figure. Each figure has a single font multiplier: **parity 1.5**, **efficiency 1.3** by default. A factor of 1 uses the base sizes; 2 doubles all text, including ticks, legends, math labels and annotations. `--font-scale 2` overrides the selected figure(s), or edit their defaults in `plot_config.yaml`. No simulation or training rerun is needed for font, spacing or legend edits. Final plots are vector PDFs with transparent backgrounds and embedded fonts.

Before drawing the band, the plot checks that its saved preprocessing checksum matches the main model. Knee diagnostics must match the learning-curve CSV checksum. The dashed marker uses validation RMSE and the prespecified [Kneedle](https://www.cs.williams.edu/~jeannie/papers/kneedle-simplex11.pdf) stability checks in the experiment protocol; unsupported knees are omitted.

## Beam inputs and data contract

The pilot keeps the original quadrupole scan of [-100, 100] for each of four strengths. Incoming alpha values vary independently by +/-20% around the stitched optics values: `alfx = 2.6283720873`, `alfy = 0.51980977991`. RMS relative momentum spread varies from 0 to 0.001 (0.1%). **These are provisional sensitivity ranges, not measured ELENA fill-to-fill tolerances.** Reference kinetic energy stays at 100 keV; beta functions and original geometric RMS emittance values stay fixed.

Alpha controls the transverse covariance, including the position-momentum correlation. Momentum spread is sampled as `delta = (p-p0)/p0`, then converted exactly to the canonical MAD-X coordinate `PT = (E-E0)/(p0*c)`. At 100 keV, `PT` is approximately `0.0146 * delta`, so feeding relative momentum spread directly into `PT` would be wrong. The stitched dispersion values are derivatives with respect to `PT`; incoming coordinates receive `[DX, DPX, DY, DPY] * PT`.

Each CSV row contains an `index`, the seven scalar inputs, and seven output columns (`mean_x/y`, `sigma_x/y`, `emittance_x/y`, `transmission`). Each output cell contains an eight-value list for start, six named observations, and end. Training selects zero-based observation 7:

- 12,000 simulations produce `X.shape == (12000, 7)` and `y.shape == (12000, 2)`.
- The seeded 80/10/10 split gives 9,600 training, 1,200 validation and 1,200 test rows when inputs are unique. Repeated complete input tuples remain in one split.
- Inputs use min-max scaling. Targets use natural log then min-max scaling. Every fitted transform uses training rows only and is saved in `preprocessing.json`; predictions are not clipped.
- The notebook architecture is retained: 7 -> 256 -> 128 -> 128 -> 32 -> 2, hidden BatchNorm/ReLU, Adam 1e-4, Huber loss, batch 32; current maximum 500 epochs with validation-based stopping.
- Historical emittance targets use uncentered second moments of canonical `x, PX` / `y, PY`. Centered final emittances are also recorded in diagnostics. Do not label these as normalized emittances or mix conventions in comparisons.

## Tracking and interpretation limits

The installed MAD-X/PTC refuses the source lattice's `MATRIX` deflectors during layout creation. The new default is therefore **native MAD-X `TRACK`**, which preserves the existing deflector matrices and thick quadrupoles. Small analytical matrix and off-momentum thick-quadrupole tests validate this path. `--backend ptc` is explicit and fails on this lattice; there is no silent fallback. This pilot is not a numerical reproduction of the paper's historical PTC dataset. A nominal, zero-spread comparison already differs in endpoint beam sizes; the investigation and missing historical provenance are documented in `results/validation/beam_input_effects/historical_comparison.md`.

Aperture checks enable broad numerical guards, not a measured hardware aperture model. Strong quadrupole settings can lose particles through numerical guards or invalid tracking domains. Surviving coordinates must all be finite, and final survivors plus recorded losses must equal the initial particle count. Diagnostics distinguish nonfinite loss records from finite guard losses. Emittances describe the surviving beam and may change when tails are removed; reported transmission is not calibrated AEgIS trapping efficiency. Neither path implements space charge.

The two primary figures follow the draft's parity and sample-efficiency layouts. Simulation variability is measured at fixed settings with independent particle draws; the blue band uses the largest observed within-setting range, following the notebook convention. It is not a universal bound or confidence interval. Learning-curve ribbons show variation across training seeds. A validation-based Kneedle marker indicates estimated diminishing returns, not a proven minimum dataset or complete saturation. See the experiment protocol for the prespecified stability checks.

## Contributors

- [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> BHARAT RAWAT](https://github.com/Bharat-1992)  
- [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> Alexander Hill](https://github.com/Alex-Hill94)  
- [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> Andrew Xu](https://github.com/Andrew-XQY)

