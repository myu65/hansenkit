# Roadmap

## M0 — Rights-clean runnable PoC

- RDKit normalization; original SMARTS features; explicit atom ownership and coverage.
- Strict molecule/polymer/EO-PO schema; refuse unsupported prediction types.
- A: chemical features + ridge (optional LightGBM); B-proxy: frozen Morgan fingerprint + ridge;
  C-proxy: chemical linear contribution + group-cross-fitted structural residual correction.
- Licensed pinned frozen MoLFormer is now available as an explicit optional local adapter;
  default Morgan path remains lightweight. Real checkpoint training on synthetic labels is recorded.
- Scaffold/polymer-family splitting, 3-component MAE/RMSE/R², extrapolation/OOD and uncertainty diagnostics.
- Cleared local CSV ingestion; deterministic synthetic CLI/CSV demonstration; tests and CI.

## M1 — Independent experimental small-molecule benchmark

- Clear a small neutral-molecule measured dataset and publish its provenance/allowed uses.
- Pre-register scaffold holdouts, units, temperature handling and calibration; compare A/B/C over seeds.
- Exact MoLFormer release review and frozen adapter are implemented; evaluate B/C on cleared measurements.
- Report teacher reproduction separately if an authorized teacher is used. Never replace real accuracy with it.
- [Opt-in six-group source-reference calculation](reference-gc.md) reproduces a new primary article's
  examples and covers 112/1,030 reference structures; it is not independent measured validation.
  Current Excel reservation leaves 11 structure candidates in 10 families, still awaiting measurement qualification.
- [Current-policy few-shot HSP reference fit](current-hsp-fewshot.md) learns those 11 records
  locally and compares 1,013 other structures. C and QM9 augmentation pass the numerical
  polar criterion only; all methods fail the dispersion/H-bond criteria. Independent measured
  noninferiority remains unproven; no calibrated interval or qualified material backend follows.

## M2 — Polymers and long-chain surfactants

- Cleared polymer measurements; repeat units, Mn, dispersity, end groups and EO/PO distribution.
- Two public experimental studies are inventoried as [28 material/API candidates](public-measurement-candidates.md);
  grades, methods and conditions remain to qualify before a material benchmark.
- Polymer-series and surfactant-family held-out evaluation, then an evidence-based scope expansion.
- Ionic surfactants and mixtures need separate physical definitions and validation before numerical output.

## M3 — Optional physics and active learning

- License-reviewed Uni-Mol2, MiniMol or CheMeleon encoder; QM9 auxiliary tasks after rights review.
- OpenMM/RadonPy backend: units, temperature, force-field coverage, charge treatment, convergence,
  cohesive energy definitions and uncertainty. No GPU jobs in routine CI.
- Active learning only after calibrated uncertainty and independent validation exist.

Do not implement all future backends in M0. Each milestone should advance one reviewable experiment.
