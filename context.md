# QuadTuneML: one-minute agent context

Updated 2026-09-08. Branch `feature/beam-alpha-momentum-spread`; previous pilot committed as `21801f8`. Current follow-up changes are uncommitted. No commit/push requested.

**Purpose:** forward neural surrogate for ELENA LNE02 / AEgIS: four quadrupole strengths plus `alfx`, `alfy`, `sigma_delta = RMS((p-p0)/p0)` predict final horizontal/vertical emittance. No control optimizer is implemented.

**Latest experiment:** `results/experiment_20260908_12000/README.md`. 12,000 settings × 6,000 particles, comprising the byte-verified previous 5,000 rows plus 7,000 new Sobol settings. Zero failed runs; all eight observations, particle accounting and 12 external source hashes pass audit. Inputs `(12000,7)`, targets `(12000,2)` select observation index 7 from legacy eight-value lists. Split 9600/1200/1200; train-only input minmax and natural-log target minmax transforms; no clipping.

**Training:** `train.py` / `train_config.yaml`, notebook MLP 256/128/128/32, BatchNorm/ReLU, Adam/Huber. Native `.venv/bin/python` (arm64 Python 3.12, TensorFlow 2.20); avoid old Intel Conda/AVX installation. Explicit seeded `tf.data` epoch reshuffling is required. Maximum 500 epochs, warm-up 100, patience 50; global best includes warm-up. Main seed 42: stopped 303, best epoch 253; validation loss 41.5% below epoch 100. Test log-scaled R² x/y **0.925/0.979**, MAE **3.76/3.18 pp**; physical R² **0.901/0.816**. Saved-model predictions independently reproduce exactly.

**Paper figures:** `plot_paper.py` is independent; `plot_config.yaml` has all-text multipliers **1.5 parity / 1.3 efficiency**. CLI `--figure parity|efficiency|both --font-scale N`; complete commands in root README. Outputs in latest run's `figures/`. `simulation_variability.py`: 64 fixed seven-input anchors × 24 particle repeats; blue bands ±1.73/4.03 pp, half the maximum observed within-anchor range, not confidence intervals. `learning_curve_diagnostics.py`: validation-only Kneedle. Thirty nested fits, seeds 42/43/44: supported vertical knee 4000, horizontal unstable; further gains remain. Nine fits hit the 500-epoch cap. All 38 tests pass.

**Physics limits:** alpha changes covariance; delta converts exactly to canonical MAD-X PT (~0.0146δ at 100 keV); entrance dispersion is per PT. Current PTC rejects MATRIX deflectors, so native TRACK preserves static matrices/thick quads. Historical endpoint disagreement remains unresolved. Numerical losses affect 2257 settings; targets use surviving particles and historical uncentered canonical second moments. No space charge, measured-beam calibration, hardware-aperture calibration or trapping-efficiency validation. Ranges remain provisional.

**Other files:** `sim_data_gen.py` / `simulation_config.yaml`; `validate_simulation.py`, `validate_experiment.py`, `tests/`. `analyze.py` retains extended analysis. Original `train.ipynb`/`config.yaml`/`utils.py` are historical four-input experiments. Prior pilot: `results/pilot_20260907/training_reshuffled/`. Generated results stay local and Git-ignored.

**Causality check:** latest run `epoch_ablation/README.md` holds data/split/seed/network fixed. Budgets 42/100/500 give test log-R² x = .833/.901/.925, y = .942/.971/.979; histories reproduce exactly. Longer training helps, but backend contribution versus the historical paper remains unisolated.
