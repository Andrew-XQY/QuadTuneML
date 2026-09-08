# QuadTuneML: one-minute agent context

Updated 2026-09-07. Work completed on `feature/beam-alpha-momentum-spread` (base `c711e97`); changes are uncommitted.

**Purpose:** forward neural surrogate for ELENA LNE02 / AEgIS: quadrupole and incoming-beam settings -> final horizontal/vertical emittance. No optimizer/control loop is implemented.

**Code map:** `sim_data_gen.py` + `simulation_config.yaml` generate data; `train.py` + `train_config.yaml` train without plotting; `analyze.py` produces seven transparent vector PDFs. `validate_simulation.py` and `tests/` verify physics and data handling. Original `train.ipynb`, `config.yaml`, and `utils.py` remain historical four-input experiments. Use native `.venv/bin/python`; requirements are pinned. Root README gives complete commands.

**Completed pilot:** `results/pilot_20260907/README.md` is the result report. Data: 5,000 settings x 6,000 particles, zero failed settings, 14 tests passed. Final model is in **`training_reshuffled/`**; figures in `figures/`. Earlier `training/` and `training_full_budget/` are diagnostic baselines. Generated artifacts are local and Git-ignored.

**Contract:** inputs are four original quad strengths plus `alfx`, `alfy`, `sigma_delta = RMS((p-p0)/p0)`: `X=(5000,7)`. Seven beam-output columns retain eight-value observation lists; select index 7 for two emittance targets, `y=(5000,2)`. Split 4000/500/500; repeated complete inputs stay together. Train-only input min-max and natural-log target min-max transforms are saved; no clipping. Notebook MLP widths 256/128/128/32, BatchNorm/ReLU, Adam/Huber. Final fits use 100 epochs / patience 100, best-validation weights. Explicit seeded per-epoch reshuffling fixes a reproduced runtime batch-order issue.

**Results:** held-out log-scaled R² x/y = 0.806 / 0.925; MAE 6.43 / 5.93 percentage points. Physical vertical-emittance R²=-1.312: tail errors remain. Single-seed sample curves improve through 4,000 training rows; no minimum established.

**Physics:** alpha changes covariance; delta converts exactly to MAD-X canonical PT (approximately 0.0146*delta at 100 keV). Entrance dispersion is per PT. Fixed original emittances/beta functions are retained. Provisional ranges: quads [-100,100], alpha +/-20%, sigma_delta [0,0.001].

**Critical limits:** current PTC rejects MATRIX deflectors, so native MAD-X TRACK preserves the current static matrices/thick quads. Analytical checks pass, but historical endpoint sizes still disagree; see validation historical_comparison.md. Source copies/hashes are saved and 12 external sources stayed unchanged. Numerical losses affect 951 settings; targets describe survivors using historical uncentered canonical second moments. No space charge, calibrated hardware aperture, trapping efficiency, or measured-beam agreement is established. Agree measured ranges and resolve historical/lattice provenance before a final physics dataset.
