# Formulation observation lookup

`formulation-lookup` returns the literal outcomes reported for a reviewed source grade and
formulation. It keeps polymer mass percentage, salt concentration/name, silica grade and mass
percentage, temperature, pH and their original units/qualifiers. The result includes the source
sheet, one-based row/column, endpoint definition, original category and evidence references.
It does not estimate an unobserved formulation, HSP or solubility.

Prepare a local `FormulationLedger` and a fresh asset-specific `FormulationReview`, then run:

```sh
uv run hansenkit formulation-lookup \
  --data local/formulations.json \
  --review local/formulation-review.json \
  --query local/formulation-query.json \
  --output local/reported-formulations.json
```

For example, this query selects the reported numeric polymer mass percentage and qualifier
for the exact source grade. These placeholder values do not identify a published material:

```json
{
  "source_grade_id": "REPLACE_WITH_EXACT_REVIEWED_GRADE",
  "conditions": {
    "polymer_wt_percent": {
      "value": 2,
      "unit": "wt%",
      "qualifier": "as_reported"
    },
    "polymer_present": true
  }
}
```

All matching records are returned. Add `endpoint_id` to choose a reviewed endpoint; omit it to
retain every endpoint. Omitted condition filters include reported values and unknowns. An
explicit null filter matches only an unreported value. A numeric filter compares the value,
declared unit and qualifier exactly; optional `unit_as_reported` also restricts the original
unit literal. Approximately reported pH does not match an `as_reported` pH query. There are no
nearest matches, ranges, concentration conversions or interpolated values.

Use `polymer_wt_percent` for mass percentage; `w/v%` is not substituted for it. The declared
units are `wt%`, `mol/L`, `degC`, `pH`, `Pa` and `h` for their respective condition fields, with
original unit text stored separately. No atmospheric pressure, waiting time, Mn, chain ends or
dispersity is supplied when the source does not report it. Nominal EO/PO counts and unspecified
mass remain nominal/source claims, rather than exact DP or Mn. Silica grade is its source label;
charge/protonation remain separate reported fields and cannot be inferred from pH.

Zero polymer concentration explicitly gives `polymer_present=false`. Those records retain the
experimental-design grade but are counted separately as controls. Repeated formulations and
conflicting outcomes remain separate records. Numeric grouping treats 22/22.0 and 0/0.0/-0.0
as the same condition without changing any returned source number, text, unit or qualifier.
Each response has a distinct source cell; two endpoints in one formulation never collapse into
one binary label. Turbidity onset or layer growth is not silently interpreted as solubility or
a cloud-point temperature.

The lookup checks fresh permission before upstream opaque-file access and ledger decoding.
Its review binds source, preparation, classification and ledger SHA-256 hashes, source-sheet
bounds, source-grade metadata, endpoint-cell/category allowlists and complete formulation/control
counts. A preparation-only permission does not authorize lookup. The disabled
[review template](../examples/formulation-review.template.json) must be completed after an
asset-specific review; it cannot open a dataset as supplied.

Datasets and generated reports remain local in ignored directories. Evaluation-only scientific,
private, noncommercial scope, original license/evidence and unknown fields survive JSON reload.
The review, ledger and every record must agree on their role: source observations use
`evaluation_only`, while authored fixtures use `authored_fixture_only`. The result retains that
same role; a fixture cannot be promoted to a source evaluation record by changing only its review.
Training, derived labels/weights, redistribution, HSP inference and physical validation remain
disabled. Own code and authored tests are MIT; they do not establish permission for source
tables or turn source categories into independent HSP measurements. No dataset is bundled.
