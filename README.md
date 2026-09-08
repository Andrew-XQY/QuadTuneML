# ELENA Extraction Beamline ML Toolkit

MAD-X simulation and a neural-network surrogate for the ELENA LNE02 transfer line to AEgIS. The current pilot predicts final horizontal and vertical emittance from four quadrupole strengths and three incoming-beam scalars: `alfx`, `alfy`, and `sigma_delta = RMS((p-p0)/p0)`.

Completed pilot: `results/pilot_20260907/README.md`. The final model is in `results/pilot_20260907/training_reshuffled/`; figures use that run.

## Current workflow

| File | Role |
| --- | --- |
| `sim_data_gen.py`, `simulation_config.yaml` | Generate deterministic seven-dimensional Sobol settings, sample particles, run isolated MAD-X jobs, and save eight-observation beam outputs. |
| `validate_simulation.py`, `tests/` | Check low-energy momentum coordinates, transverse covariance, dispersion, tracking maps and preprocessing. |
| `train.py`, `train_config.yaml` | Standalone notebook MLP; save split, fitted transforms, model, predictions, metrics and optional sample-size sweep. |
| `analyze.py`, `APS_PLOT_STYLE.py` | Independent data checks and transparent vector PDF figures. Does not train or simulate. |
| `train.ipynb`, `config.yaml`, `utils.py` | Original four-input notebook experiments, retained for reference. |
| `context.md` | One-minute handoff for the next agent. |
| `results/pilot_20260907/` | New pilot data, model, diagnostics, figures and run summary. Generated artifacts remain local and Git-ignored. |

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
  --samples 5000 --particles 6000 --workers 8 \
  --output results/new_pilot/data.csv

.venv/bin/python train.py --data results/new_pilot/data.csv \
  --output-dir results/new_pilot/training --threads 2 --epochs 100 --patience 100 \
  --sample-efficiency 500,1000,2000,3000,4000

.venv/bin/python analyze.py --data results/new_pilot/data.csv \
  --training-dir results/new_pilot/training \
  --output-dir results/new_pilot/figures
```

The pilot commands give every fit the full 100-epoch budget and select the best validation checkpoint. This avoids premature early stopping during BatchNorm warm-up in small subsets; the configuration file retains the original patience-5 default for comparison. Training uses explicit seeded per-epoch reshuffling, verified to change batch order across epochs while reproducing a fresh run.

Training and figure generation can be rerun independently of simulation. Training refuses to overwrite an existing model. Simulation exits with a nonzero status if any setting fails; inspect `metadata.json`, `diagnostics.csv`, `failures.jsonl`, and `failed_cases/` before using an incomplete batch.

## Beam inputs and data contract

The pilot keeps the original quadrupole scan of [-100, 100] for each of four strengths. Incoming alpha values vary independently by +/-20% around the stitched optics values: `alfx = 2.6283720873`, `alfy = 0.51980977991`. RMS relative momentum spread varies from 0 to 0.001 (0.1%). **These are provisional sensitivity ranges, not measured ELENA fill-to-fill tolerances.** Reference kinetic energy stays at 100 keV; beta functions and original geometric RMS emittance values stay fixed.

Alpha controls the transverse covariance, including the position-momentum correlation. Momentum spread is sampled as `delta = (p-p0)/p0`, then converted exactly to the canonical MAD-X coordinate `PT = (E-E0)/(p0*c)`. At 100 keV, `PT` is approximately `0.0146 * delta`, so feeding relative momentum spread directly into `PT` would be wrong. The stitched dispersion values are derivatives with respect to `PT`; incoming coordinates receive `[DX, DPX, DY, DPY] * PT`.

Each CSV row contains an `index`, the seven scalar inputs, and seven output columns (`mean_x/y`, `sigma_x/y`, `emittance_x/y`, `transmission`). Each output cell contains an eight-value list for start, six named observations, and end. Training selects zero-based observation 7:

- 5,000 simulations produce `X.shape == (5000, 7)` and `y.shape == (5000, 2)`.
- The seeded 80/10/10 split gives 4,000 training, 500 validation and 500 test rows when inputs are unique. Repeated complete input tuples remain in one split.
- Inputs use min-max scaling. Targets use natural log then min-max scaling. Every fitted transform uses training rows only and is saved in `preprocessing.json`; predictions are not clipped.
- The notebook architecture is retained: 7 -> 256 -> 128 -> 128 -> 32 -> 2, hidden BatchNorm/ReLU, Adam 1e-4, Huber loss, batch 32, 100 epochs; the original patience-5 default can be overridden as in the pilot commands above.
- Historical emittance targets use uncentered second moments of canonical `x, PX` / `y, PY`. Centered final emittances are also recorded in diagnostics. Do not label these as normalized emittances or mix conventions in comparisons.

## Tracking and interpretation limits

The installed MAD-X/PTC refuses the source lattice's `MATRIX` deflectors during layout creation. The new default is therefore **native MAD-X `TRACK`**, which preserves the existing deflector matrices and thick quadrupoles. Small analytical matrix and off-momentum thick-quadrupole tests validate this path. `--backend ptc` is explicit and fails on this lattice; there is no silent fallback. This pilot is not a numerical reproduction of the paper's historical PTC dataset. A nominal, zero-spread comparison already differs in endpoint beam sizes; the investigation and missing historical provenance are documented in `results/validation/beam_input_effects/historical_comparison.md`.

Aperture checks enable broad numerical guards, not a measured hardware aperture model. Strong quadrupole settings can lose particles through numerical guards or invalid tracking domains. Surviving coordinates must all be finite, and final survivors plus recorded losses must equal the initial particle count. Diagnostics distinguish nonfinite loss records from finite guard losses. Emittances describe the surviving beam and may change when tails are removed; reported transmission is not calibrated AEgIS trapping efficiency. Neither path implements space charge.

The figure layout follows the draft's ML parity and sample-efficiency plots. The sweep uses fresh models, nested training subsets, fixed holdouts and a common reference scale for displayed errors. A 5,000-total-sample, single-seed seven-input pilot does not establish a new minimum sample requirement or inherit the paper's old threshold. No stochastic-variability band is drawn without a matching repeated-run study.

## Contributors

- [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> BHARAT RAWAT](https://github.com/Bharat-1992)  
- [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> Alexander Hill](https://github.com/Alex-Hill94)  
- [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> Andrew Xu](https://github.com/Andrew-XQY)

