# Evaluation contract and first synthetic run

There are three distinct evaluation bases: `synthetic`, `teacher_reproduction` and `experimental`.
Each dataset has one label kind; its CSV rows must match its checksum-bound manifest.
Teacher labels require teacher identity and derived-label permission. Experimental data require
independent measurements, original provenance and specific permission. No real or external
teacher data is shipped. A teacher-trained model may be tested against independent measurements
in a **separate report**, with both training and evaluation label kinds recorded.

## Splits and metrics

Canonical duplicates and scaffold groups remain within one of training, calibration or test.
All acyclic compounds share a conservative `ACYCLIC` family. We do not fall back to random
splits when few families exist. At least eight independent groups are required. A polymer-series
strategy requires every row to carry a series ID, joins conflicting IDs through duplicate connectivity,
and isolates families. Actual polymer training is still blocked by scope; the splitter can be used
for cleared representative small-molecule/oligomer family experiments.

Default proportions are approximately 60/20/20 of groups, not a guarantee of those row proportions.
MAE/RMSE/R² are reported for δD/δP/δH in MPa^0.5. R² is null for constant or singleton truth.
External holdout evaluation refuses canonical and scaffold/family overlaps with training.

OOD is a diagnostic threshold (nearest training Morgan Tanimoto <0.35), not a validated domain
classifier. Extrapolation means a chemical feature lies beyond a training feature range. Reports
contain separate OOD/non-OOD and extrapolation/non-extrapolation metrics, including empty
subset markers. The threshold was fixed without optimizing against test results.

Uncertainty uses split conformal calibration with one maximum absolute error per calibration
group/component and a finite-sample order statistic. The default nominal component group
coverage is 80%; components are not jointly calibrated. Too few calibration groups for the requested
coverage yields no finite interval rather than false precision. Reports show empirical per-row and
per-group coverage and interval widths on held-out test groups. Exchangeability is required for a
coverage guarantee and is doubtful under scaffold/OOD shift. Intervals on synthetic labels do not
establish physical confidence. Uncertainty is not yet a basis for active acquisition.

## Recorded smoke experiment (2026-10-10)

Windows, Python 3.12, committed `uv.lock`, seed 42, **356 original synthetic rows**,
29 scaffold/acyclic groups: train 224 rows /17 groups, calibration 66 /6, test 66 /6.
The targets were generated mainly from A's chemical features plus arbitrary nonlinear terms/noise;
this intentionally favors A. This experiment verifies plumbing, **not physical accuracy or which
architecture will win on measurements**. No literature coefficients or HSP teacher were used.

| Runnable path | δD MAE / RMSE / R² | δP MAE / RMSE / R² | δH MAE / RMSE / R² |
| --- | --- | --- | --- |
| A ridge | 0.0756 / 0.0926 / 0.9869 | 0.1522 / 0.1967 / 0.9678 | 0.0793 / 0.0948 / 0.9912 |
| B fingerprint proxy | 0.3428 / 0.3915 / 0.7655 | 0.9380 / 1.2841 / -0.3715 | 0.7754 / 1.0687 / -0.1236 |
| C residual proxy | 0.0887 / 0.1185 / 0.9785 | 0.4583 / 0.6667 / 0.6303 | 0.1627 / 0.2314 / 0.9473 |
| A LightGBM | 0.0901 / 0.1086 / 0.9820 | 0.1396 / 0.2198 / 0.9598 | 0.0893 / 0.1310 / 0.9831 |

The hybrid did not improve this synthetic case. Do not tune the PoC to reverse that result.
A-ridge empirical group coverage was 1.000 /0.667 /1.000 despite nominal 0.8, illustrating why
coverage must be measured and why six test/calibration groups cannot establish dependable uncertainty.
The full reports are reproducible from the README demo and stay local; no trained weights are published.

## First substantive experiment

Start [#2](https://github.com/myu65/hansenkit/issues/2) and [#4](https://github.com/myu65/hansenkit/issues/4):
clear original neutral-molecule measurements with units/temperature and permitted uses, freeze
scaffold holdouts before fitting, and run A ridge/LightGBM over several seeds. Review distributions,
OOD errors and coverage. Then finish the exact MoLFormer review in [#3](https://github.com/myu65/hansenkit/issues/3)
and compare B/C on the same holdouts. Decide whether embeddings help using those measurements.
Begin polymer/EO-PO experiments only after this baseline and series data permissions are established.
