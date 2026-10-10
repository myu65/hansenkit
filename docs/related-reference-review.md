# Related calculations and reference data

Reviewed locally on 2026-10-10 under the user's local audit assumption. No external tables,
checkpoint files, publisher models, fitted heads or per-compound predictions are distributed.
Numerical reports and source checksums remain local. This review does not mark the full goal achieved.

## Published HSP predictors

[HSP-predictions](https://github.com/darjacvetkovic/HSP-predictions) at
`93f494824bca592d24f777db1dc6f975149e0aff` includes native JSON XGBoost models with
10/30/50 descriptors. All nine models were loaded locally with XGBoost 3.4.1, using their exact
feature names and the provided descriptor table. No pickle or notebook execution was required.
All three models produced finite predictions on the same 1,030-compound reference population.

| Publisher model | δD MAE | δP MAE | δH MAE | Role |
| --- | ---: | ---: | ---: | --- |
| 10 descriptors | 0.470 | 1.411 | 1.166 | Reference reproduction diagnostic |
| 30 descriptors | 0.253 | 1.578 | 0.738 | Reference reproduction diagnostic |
| 50 descriptors | 0.350 | 1.199 | 0.981 | Reference reproduction diagnostic |

Units: MPa^0.5. **These are not independent accuracy comparisons to our held-out models.**
The publisher notebook partitions the same reference source twice with `random_state=42`.
Reconstructing that recipe yields 762 training / 191 validation / 239 test rows, containing
16 / 2 / 9 Excel identity rows respectively. Canonical duplicate overlap is 4 between train and
validation, 6 between train and test, and 3 between validation and test. The weights do not
serialize exact sample membership, so this is a verified recipe reconstruction, not proof of
every released weight's training IDs. Source-population overlap is sufficient to disqualify
the full-population score as an independent held-out test.

Do not distill these models into a student claimed to be independent of the reserved Excel.
A teacher-reproduction experiment needs its own metrics and artifacts. A fair algorithm
comparison must refit descriptor selection, imputation and regression on our training groups;
the publisher's supervised selected-feature lists cannot be treated as independently selected
for a test population drawn from the same labels.

## Reference tables

[HSPiPy](https://github.com/Gnpd/HSPiPy) was inspected through CSV files, not pickle objects.
Its `db.csv` has 249 rows: 234 rows overlap HSP_SMILES identities, with 159 triples matching
within 0.051 MPa^0.5 and 75 differing. The example table has 1,206 rows: 1,093 overlap, with
1,077 matching and 16 differing. The threshold identifies differences, not an assertion that
every difference is physically significant or incorrect. Record original source, temperature,
rounding, units and measured/estimated status before resolving disagreements.
Shared values do not provide independent confirmation, and duplicate tables do not multiply
the number of distinct labeled compounds. Do not merge conflicting labels blindly.

[PolySol](https://github.com/cstubb/PolySol) at `5d2fc43be8f547acdbe2441b805969e653244d62`
contains 1,818 homopolymer/solvent and 270 copolymer/solvent records. Its published fold CSV
also provides predicted probabilities and train/validation/test markers. These are solvent
compatibility classes, not three HSP target values. The structure fields represent monomers;
they must not be mislabeled as chemically assembled polymer repeat units. This dataset can
test polymer–solvent compatibility as a separate endpoint, with polymer-series isolation and
correct polymerization mapping, after its source/condition metadata are retained.

The [Mendeley polymer parameter dataset](https://data.mendeley.com/datasets/f9gs23sg6y/1)
is another candidate. Its listing declares CC BY 4.0 and a workflow file, but the data files
have not yet been inspected locally. Do not assume scalar Hildebrand parameters or binary
solubility measurements are three-component HSP labels.

## Calculators require formula and atom-count checks

[early-screening-des](https://github.com/iehoshva/early-screening-des) at
`1c65846c49246d8e13c46175b2fd4c5ebdc89651` was inspected without executing its batch workflow.
Its declared `F_pi²` entries for alcohol/ether resemble conventional unsquared force contributions;
an alcohol unit check raises a force-square/unit consistency concern. Units and the original
coefficient source need confirmation before adopting it. Generic unsaturated-carbon rules also precede composite
carbonyl groups. The code explicitly notes unrepresented ionic atoms. This is a diagnostic
candidate, not a validated fallback for ions or large molecules.

[Mathieu's primary paper](https://doi.org/10.1021/acsomega.8b02601) supplies a spreadsheet of
calculated/reference values and a Python implementation. It distinguishes measured and
estimated validation values and discusses limitations for polar groups pointing in opposite
directions. Its supplementary assets are a priority for an equation-based comparator; they
were obtained locally through the publisher's public download controls. The unchanged program
has been run on the current reference structures, and input identity errors were found.
[Source lineage, common-row comparison and CAS audit](mathieu-reference-audit.md) record
the evidence. Avoid substituting a loosely implemented repository calculator for the
published method without checking equations, fragment counts and coverage.

## Uncertainty result from our initial experiment

The 1,030 distinct compounds span 121 scaffold groups. Every row was held out once; none of
the reserved Excel identities or scaffolds entered a supervised fit. Training-only Morgan
similarity at threshold 0.35 marks 792 OOD rows. With 10,000 paired scaffold-bootstrap resamples,
A and A+quantum pass the point MAE/RMSE ratio criterion against the current HSPiT code.
Their δP one-sided 95% upper MAE differences are 0.407 and 0.410 MPa^0.5, exceeding the
predeclared 0.30 bound. **No candidate passes the complete noninferiority criteria.**
The comparator's equation/counting validity and independent measurement lineage remain
unresolved, and predictive intervals for these exploratory heads are not calibrated.

The next experiment must retain this negative result and all candidates, improve polar-feature
physics and train-only selection, verify the published comparator, and report any reuse of this
development benchmark. Reserve a genuinely new evaluation population before final model
selection. Polymer and long-chain numerical/physical validation remain part of the active goal.
