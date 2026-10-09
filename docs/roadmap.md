# Roadmap

## M0 — Rights-clean runnable PoC

- RDKit normalization; original SMARTS features; explicit atom ownership and coverage.
- Strict molecule/polymer/EO-PO schema; refuse unsupported prediction types.
- A: chemical features + ridge (optional LightGBM); B-proxy: frozen Morgan fingerprint + ridge;
  C-proxy: chemical linear contribution + group-cross-fitted structural residual correction.
- Encoder interface for future licensed, fixed MoLFormer embeddings, disabled by default.
- Scaffold/polymer-family splitting, 3-component MAE/RMSE/R², extrapolation/OOD and uncertainty diagnostics.
- Cleared local CSV ingestion; deterministic synthetic CLI/CSV demonstration; tests and CI.

## M1 — Independent experimental small-molecule benchmark

- Clear a small neutral-molecule measured dataset and publish its provenance/allowed uses.
- Pre-register scaffold holdouts, units, temperature handling and calibration; compare A/B/C over seeds.
- Review exact MoLFormer-XL-both-10pct revision, weights and complete dependencies; implement frozen adapter.
- Report teacher reproduction separately if an authorized teacher is used. Never replace real accuracy with it.

## M2 — Polymers and long-chain surfactants

- Cleared polymer measurements; repeat units, Mn, dispersity, end groups and EO/PO distribution.
- Polymer-series and surfactant-family held-out evaluation, then an evidence-based scope expansion.
- Ionic surfactants and mixtures need separate physical definitions and validation before numerical output.

## M3 — Optional physics and active learning

- License-reviewed Uni-Mol2, MiniMol or CheMeleon encoder; QM9 auxiliary tasks after rights review.
- OpenMM/RadonPy backend: units, temperature, force-field coverage, charge treatment, convergence,
  cohesive energy definitions and uncertainty. No GPU jobs in routine CI.
- Active learning only after calibrated uncertainty and independent validation exist.

Do not implement all future backends in M0. Each milestone should advance one reviewable experiment.
