# Project instructions

Build a small, runnable, HSPiP-independent Python research tool. Own code is MIT.
Prefer one focused issue and PR at a time; avoid a service framework or a plugin marketplace.
Read README.md and docs/data-policy.md before changing data ingestion or model training.

## Non-negotiable scientific and rights boundaries

- Never import HSPiP coefficients, HSPiT Excel tables, HSP_SMILES.csv, proprietary company data,
  noncommercial weights, or derived pseudo-labels without asset-specific permission evidence.
  A repository's code license alone is not clearance for its tables, data, or weights.
- Keep synthetic, teacher and experimental training labels in separate datasets/model provenance.
  A teacher-trained model may be evaluated on independent measurements in a separate report
  recording both label kinds. Never describe synthetic/teacher agreement as real accuracy.
- Keep local data/models in ignored directories. Do not upload them or log raw private records.
- Normalize before grouping. Keep canonical duplicates and scaffold/polymer families within one split.
  Preserve legacy Murcko links and the current conservative core/complete bounded tautomer aliases.
  Grouping keys never replace encoder inputs; partial enumeration cannot certify isolation.
  Reaudit historical fits and refit conflicts; never repair contaminated training by filtering test only.
  Fit preprocessing, residual models, OOD references, and uncertainty using training/calibration only.
- Reject unsupported polymers, mixtures, ionic molecules and EO/PO surfactants during inference.
  Schema support is not a scientific applicability claim. No guessed HSP fallback.
- Pretrained encoders are off until weights, code, dependencies, exact revision, and execution
  permissions are documented. Do not silently download weights or execute remote model code.
  The audited optional MoLFormer release can run through the explicit local checkpoint/review
  flags; read docs/molformer-review.md. Never broaden its immutable code/weight allowlist casually.
- The user assumes audit passage for the 2026-10-10 local research experiment. Record this as
  `audit_basis="operator_assumption"`; it does not authorize public data/model redistribution.
  HSPiT Excel labels/coefficients remain evaluation-only and cannot supply training targets,
  auxiliary targets, fitted features, tuning or calibration. Reserve identities/families before
  fitting, including auxiliary QM9 learning. Published reference values with unknown measurement
  lineage use `published_reference`, never `experimental`.
- Research extensions may evaluate neutral large molecules and assembled nonionic polymers/
  surfactants through explicitly qualified backends. Preserve default PoC refusals until a
  backend's own applicable scope and validation evidence are implemented. Never invent a
  physical accuracy claim from stable finite arithmetic or auxiliary/teacher training counts.
- `reference-gc` is a separately labeled local coefficient-reference calculation, not a qualified
  physical backend. Read docs/reference-gc.md; keep coefficient files out of Git and preserve
  hash-bound local-use reviews, complete atom coverage, bounded size and explicit caps.
  Never pass its outputs into training, pseudo-label datasets or the default HSP predictor.

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
