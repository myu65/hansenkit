# Project instructions

Build a small, runnable, HSPiP-independent Python research tool. Own code is MIT.
Prefer one focused issue and PR at a time; avoid a service framework or a plugin marketplace.
Read README.md and docs/data-policy.md before changing data ingestion or model training.

## Non-negotiable scientific and rights boundaries

- Never import HSPiP coefficients, HSPiT Excel tables, HSP_SMILES.csv, proprietary company data,
  noncommercial weights, or derived pseudo-labels without asset-specific permission evidence.
  A repository's code license alone is not clearance for its tables, data, or weights.
- Synthetic demos, teacher reproduction, and independent experimental evaluation are separate
  label kinds, datasets, models, and report sections. Never describe synthetic/teacher metrics as real accuracy.
- Keep local data/models in ignored directories. Do not upload them or log raw private records.
- Normalize before grouping. Keep canonical duplicates and scaffold/polymer families within one split.
  Fit preprocessing, residual models, OOD references, and uncertainty using training/calibration only.
- Reject unsupported polymers, mixtures, ionic molecules and EO/PO surfactants during inference.
  Schema support is not a scientific applicability claim. No guessed HSP fallback.
- Pretrained encoders are off until weights, code, dependencies, exact revision, and execution
  permissions are documented. Do not silently download weights or execute remote model code.

## Development

- Python 3.11/3.12, src layout; use uv and the committed lockfile.
- After behavioral changes run `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run pytest`, and the documented synthetic CLI demo. Build the distribution for packaging changes.
- Add meaningful tests for data gates, atom ownership/coverage, split isolation, refusals,
  model round trips and the end-to-end CLI. Do not tune on test results.
- PRs describe the trigger, resulting behavior, checks, remaining scientific limitations, and issue.
  Merge after CI passes when the user has authorized it. Do not change unrelated repositories or settings.
- Keep model/physics extension interfaces small. New encoder/backend implementations need their
  own rights review and tests. No subagent delegation is required by this project.

## Code Review Rules

Flag any leakage, rights-gate bypass, implicit external downloads, unsupported numeric predictions,
or synthetic results presented as physical validation. Check artifact provenance survives reload.
