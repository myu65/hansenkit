# HSPiT Excel inspection and evaluation reservation

The user explicitly requested local inspection/classification on 2026-10-10. The inspected
[upstream revision](https://github.com/Emil-Kongsbach/HSPiT/tree/84ebcc897f62f660c28d4f16ebf788fdf0e4e330)
is fixed. No workbook, numerical coefficients, extracted HSP labels or public pseudo-labels are
included here. Tables were not used for training, calibration, model selection or accuracy claims.

| Workbook | Observed structure | Evaluation classification |
| --- | --- | --- |
| `stefanis_data2.xlsx` | Four sheets: `first_order` 77×9, `second_order` 38×6, `low_first_order` 44×5, `low_second_order` 12×4. Columns describe groups, dd/dp/dh contributions and SMARTS. Dimensions include headers. | Group-contribution coefficient tables, not molecule-level independent measured labels. A permitted future method comparison would measure teacher/reference agreement. |
| `fitting_data.xlsx` | `Sheet1` 28×9 including header; 27 solvent rows. Columns `solvent,dd,dp,dh,score,red,mvol,cas,smiles`. One unusable SMILES at Excel row 17. | Molecule-level HSP reference table used as solvent inputs by upstream hsp_fit.py. Potential evaluation identities; original measurement source, temperature, independence and data permission are unverified. |

Inspected SHA-256 values:

- `stefanis_data2.xlsx`: `164b98ec71e00ea84d8900cb5ab591767ce4f76cd75aa619472dfa3204b22130`
- `fitting_data.xlsx`: `816a7c4302b70c8c04d3bd7a57008896eb37cd9f52a13c6e26750e1a11a79094`

MIT for repository code does not alone settle third-party data rights or establish independent
measurements. Asset review remains [issue #2](https://github.com/myu65/hansenkit/issues/2).
Authorized structural inspection differs from numerical evaluation/training/redistribution permission.

## Preserve a future true holdout

Before the new MoLFormer regression fits, 26 parseable solvent identities and their nine scaffold/
acyclic keys were reserved. Excluding matching keys removed 103 of 356 original synthetic rows.
The remaining 253 rows share no molecule/scaffold with usable reference identities: 143 training
/13 groups, 55 calibration /5, 55 test /5. Only identity membership was consulted; no HSP number
or group coefficient entered the generator, fitted targets, calibration or model selection.
This is supervised isolation; encoder pretraining has not been decontaminated against the table.

Earlier 356-row smoke models are engineering controls and are not eligible for this reserved
external holdout. New local artifacts remain synthetic-only. After rights and measurement
provenance are cleared, freeze an evaluation-only dataset manifest (`allowed_role="evaluation_only"`).
Training with this role is rejected even if training permission flags are accidentally set to true.

Next evidence: original source and permitted local evaluation; measured versus estimated lineage;
units/temperature; CAS/SMILES resolution; independent measurement status. No public trained HSP
model should inherit table labels without that evidence and derived-weight/redistribution permission.
