# Measured-property auxiliary source review

The NIST ThermoML archive supplies thermophysical properties, not Hansen component
labels. This local review adds **zero HSP training or independent HSP test triples**.
Density and vaporization enthalpy remain separate auxiliary tasks; they cannot
uniquely determine dispersion, polar and hydrogen-bond contributions.

## Fixed source and integrity

The [official archive](https://doi.org/10.18434/mds2-2422), snapshot 2020-09-30,
contains cooperating-journal articles through 2019. The downloaded gzip archive
matches the publisher's 189,433,115 bytes and SHA-256
`231161b5e443dc1ae0e5da8429d86a88474cb722016e5b790817bb31c58d7ec2`.
Both inspected metadata revisions identify these same bytes. A newer metadata
revision is not a new measurement population.

Own read-only code scanned 11,923 article JSON members without extracting the tar
or running upstream programs. The source property counts were reproduced exactly:

| Pure-component archive property | Values | Source property blocks | DOIs |
| --- | ---: | ---: | ---: |
| Mass density | 182,802 | 11,956 | 3,465 |
| Amount density | 1,961 | 33 | 15 |
| Vaporization or sublimation enthalpy | 1,008 | 903 | 221 |

Values, source blocks, unique compounds, molecular families and independent assays
are different counts. The combined enthalpy records comprise 678 liquid-vaporization
records, 326 crystal-sublimation records and four unresolved transitions. All 1,008
lack explicit pressure in the corresponding archive datasets; no pressure is filled in.

## Identity, state and isolation

The review preserves original targets and uncertainties, verifies Standard InChI
against the source InChIKey, and checks neutral molecular scope, complete bounded
tautomer grouping, component/solvent records, phase, method and explicit conditions.
All 2,798 existing reservation aliases and the additional seven HSP training
identities/families are excluded. An identity-only check also found no overlap with
the previous eleven HSP training families. No HSP numeric table was read for this task.

After metadata filtering, 287 mass-density points remained as candidates. Primary
state review held 28 points with unresolved solution interpretation and eleven
below the same-source normal melting point. The retained **248 density points /
21 graphs / 16 conservative families / 31 DOIs** have explicit temperature and
pressure, one Component, no separate Solvent or composition variable, a Liquid
phase, direct mass-density presentation and a declared measurement method.
Every retained record matches its exact archive source and JSON pointer.
These are 174 exact graph/temperature/pressure conditions; the other 74 source
points repeat such conditions. Values and source/sample lineage are retained
without treating those repeats as independent molecular examples.

Their qualification is `measured_pure_liquid_as_archived_pending_primary_table_validation`.
This confirms the archive record, not an independent match to the authors' full
numeric tables, stable equilibrium, sample identity under all conditions, or absence
of archival errors. Ambient density must not be censored using a high-pressure
melting point; that review comparison was corrected before any fitting. Prior
reviews and raw records remain private and unchanged.

Five calorimetry-tagged vaporization records in five graphs/four families remain
separate candidates with unknown pressure. There is **no shared graph** between
the retained density and enthalpy subsets. They do not provide a matched-state
molar-volume/cohesive-energy dataset. The archive omits internal TRC erroneous or
corrected-point flags; consult the [primary archive paper](https://doi.org/10.1002/jcc.26842)
and original methods before a stronger accuracy claim.

## Local experiment boundary

Preparation permissions remain `trainingAllowed=false` and
`publicRedistributionAllowed=false`. A separately frozen, checksum-bound local
fitting review is required for any auxiliary pilot. Keep original-unit metrics,
temperature/pressure, method, expanded-uncertainty conventions, exact graphs,
families and source DOIs. All conditions and duplicates of a molecular family
must stay in one split; fit scalers, residuals and OOD references inside training only.

The separately reviewed local pilot evaluates density generalization by held-out molecular
families. It does not refit the exposed 1,013-structure HSP evaluation or qualify
polymers, ionic surfactants, mixtures or a default HSP backend. Repeated temperatures
increase rows, not independent molecule/family counts; this isolated subset remains
below the 10,000-example learning target.

## Local family-held-out pilot

All 16 conservative families were held out once. Training uses unweighted
source-point natural-log density, fixed ridge alpha 100, and training-only scaling.
A adds 105 chemical features to explicit T/P; B adds the pinned 768-dimensional
fixed MoLFormer CPU embedding; C learns residuals from four inner training-family
folds before predicting the outer held-out family. Constant and condition-only
baselines use those same outer partitions. There was no test-driven selection or
subsequent all-data inference-model fit.

| Model | Point MAE | Equal-graph MAE | Equal-family MAE | Equal-family RMSE | Equal-family R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant | 84.05 | 105.09 | 117.54 | 140.62 | -0.184 |
| T/P only | 95.83 | 110.55 | 124.77 | 147.19 | -0.297 |
| A: chemical | 43.84 | 51.58 | 59.18 | 78.06 | 0.635 |
| B: fixed MoLFormer | 38.40 | 59.12 | 67.77 | 92.37 | 0.489 |
| C: cross-fitted residual | 56.82 | 56.47 | 58.61 | 75.81 | 0.656 |

Error units are kg/m³. Point metrics weight each archived source point equally;
graph/family metrics give each respective group equal total weight, preserving
all individual source values. B has the lowest point MAE, but A has the lowest
equal-graph MAE; the small A/C equal-family difference establishes no reliable
hybrid advantage. Neither different weighting nor repeated source conditions
supplies more independent molecular examples.

There are 155/248 chemistry-envelope OOD points and 40/248 T/P-range OOD points,
assessed inside the respective outer training partitions; these counts can overlap.
Raw OOF scores are research diagnostics. The scope helper refuses extrapolating
inference, and no calibrated prediction intervals or default HSP/material outputs
follow from this small as-archived pilot. Shared source DOIs across different families
are reported rather than asserted to be independent assays.

The first private scalar-head execution stopped before producing saved validation
predictions or scores. A shape-only repair passed actual synthetic scalar fits,
independent sklearn replay and JSON round trips; exact existing CPU features were
reused after source, order, folds, encoder and module parity checks. Original files
and statistical choices were retained.

The private final plan SHA-256 is
`71471195adacde6408c7860abb38979c6184340f7b08630f9d6389f397e56c91`;
fixed OOF prediction SHA-256 is
`d6127767f6a8dc62dd513c55a5b8015f9da0aee6c58993a223264cecc358c054`;
evaluation SHA-256 is
`0bf94997e8c173a355122a5e86ca611f9ed1e01b9b9d8c8ca0e23d2f2a04b6f4`.
Arrays, source rows, features and fold-model states remain private. Independent
sklearn replay reproduced all 16 outer and 64 inner folds, C residuals and saved
heads with a maximum prediction difference of 1.14e-12 kg/m³; JSON restoration
was exact. Independent NumPy group moments reproduced all reported metrics,
the 155/40 OOD counts and per-fold DOI overlaps. The verification receipt SHA-256 is
`4ebdad3c8be80828272703b3b6613a121db80abc39d56b39d26722bb95a48067`.
Source and prior artifacts remained unchanged; no production refit, encoder
re-extraction, HSP data read or model selection was performed by the verifier.

The metadata declares [NIST terms](https://www.nist.gov/open/license), with separate
SRD and other-work provisions; it is not an MIT or CC0 declaration. The
[archive description](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive)
states that the journal files are available with publisher permission. Preserve
that evidence, attribution and every upstream DOI. The user's scoped local
`audit_basis="operator_assumption"` does not authorize publishing source records
or derived model weights. Only this original aggregate review is included in Git.
