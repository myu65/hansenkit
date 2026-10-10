# Fixed-coordinate source-reference solubility experiment

This is a separate **log10(mol/L)** auxiliary task with zero HSP training triples.
No HSP/Excel numerical value, coefficient, prediction or pseudo-label supplies
training, features, calibration or tuning. It qualifies no molecular/polymer/
surfactant HSP backend. Default HSP applicability refusals remain in place.

## Source qualification

The [author BigSolDB v2.1 release](https://huggingface.co/datasets/levakrasnov/BigSolDBv2.1)
at `8d7442c123d417fe4c18fd91524c176ede359f27` declares CC BY 4.0. The 112,465-row CSV
passes publisher LFS SHA-256
`212cb511e05051917f7ab8811282c06c99512e4f720cf270be555fdf07b4a854`.
The [primary methods paper](https://www.nature.com/articles/s41597-025-05559-8.pdf)
describes literature mole-fraction measurements and density-based molar-concentration
conversion, including interpolated solvent densities. Individual phase, solid form,
sample stereochemistry and conversion density remain uncertified. Thus the label
kind is `published_reference_solubility_auxiliary`, not measured HSP truth.

All 526 available candidate PubChem CIDs were independently retrieved. 506 source
graphs exactly match normalized **isomeric** public SMILES; 20 mismatches remain
unresolved, and 37 earlier candidates lack a valid CID. No source graph is replaced.
The subset also requires valid CAS checksum/DOI syntax, a supported neutral connected
solute, completed current grouping, a pure connected neutral fixed solvent context,
finite positive temperature K and finite logS. These checks do not certify a sample.

| Sequential rejection or retention category | Rows |
| --- | ---: |
| Reserved HSP identity/family/core | 49,164 |
| Unsupported charge, fragment, radical or dummy | 24,387 |
| Incomplete tautomer grouping | 539 |
| Invalid or missing CAS | 98 |
| Independent graph mismatch | 1,426 |
| Independent CID missing | 1,098 |
| Nonorganic solute | 81 |
| Unsupported fixed solvent context | 176 |
| Missing or nonfinite numeric target | 847 |
| Retained | 34,649 |

The categories reconcile every source row. The retained population contains 502
solutes, 336 groups and 66 fixed coordinates. Conflicting same-context measurements
are retained; only exact complete source-record duplicates could collapse. No
clipping, zero-filling of missing labels, temperature interpolation or error-based
exclusion is performed. Source records/traces remain local.

Original acquisition/preparation fitting flags remain off. A separate
operator-assumption review binds the exact subset and immutable plan for local
fitting/derived states, with no external-data or model redistribution. No upstream
EnSol code, pickle or weights are executed.

## Fixed coordinates and family isolation

Solvent graphs are coordinate evidence only, never encoder or regression inputs.
Each named solvent is a fixed target context, analogous to a fixed solvent-energy
column in [SolQuest](solvation-auxiliary.md). A coordinate-specific head predicts
logS for a **solute** at K in that already specified context. This does not authorize
trainable molecular pair encoders, global solvent embeddings, unseen-solvent
inference, a solvent HSP head or mixtures. No numerical solvent HSP is supplied.

Solute aliases use `canonical+strict-murcko+core-topology+bounded-tautomer-v1`.
All temperatures/solvents of each solute family remain in one partition, with
zero HSP-reservation and partition alias overlap. Observations are not randomly split.
Prospective structural/presence support rules select 22 coordinates: at least eight
training groups/16 training solutes, nine calibration groups and five test groups.

| Partition across selected coordinates | Observations | Solutes | Groups |
| --- | ---: | ---: | ---: |
| Training | 17,129 | 263 | 199 |
| Calibration | 6,695 | 118 | 67 |
| Test | 5,242 | 115 | 67 |

These are 17,129 training records, not 17,129 independent compounds or HSP triples.

## Frozen A/B/C comparison

A uses the existing 105 original chemical-density/count features; B uses reviewed
frozen 768-dimensional MoLFormer 10% embeddings; C adds a four-fold whole-family
cross-fitted chemical residual embedding head. Group coverage is incomplete in 238
of 502 descriptor records and explicitly reported, without invented groups.
CPU extraction does not update the encoder.

For molecular vector `v`, the design is
`[v, v * (1000/T_K - 1000/298.15), 1000/T_K - 1000/298.15]`.
It is a statistical temperature input, not inferred enthalpy or a phase model.
Ridge alpha 100 and scaling use training only. Inverse observation counts per
training solute, normalized to mean one, prevent heavily measured compounds from
dominating; inner folds calculate their own weights/scaler. Test errors do not
select/refit a model. Source values were previously inventoried; no blinded HSP
benchmark or independent experimental HSP accuracy claim is made.

| Method | Macro MAE | Macro RMSE | Mean R2 | Solute-balanced MAE | Negative R2 coordinates |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, chemical | 0.727 | 0.944 | 0.089 | 0.764 | 6 / 22 |
| B, frozen MoLFormer | 0.944 | 1.186 | -0.514 | 0.975 | 18 / 22 |
| C, cross-fitted residual | 1.043 | 1.333 | -1.001 | 1.043 | 21 / 22 |

Error units are log10(mol/L). Macro scores average coordinates equally; solute-balanced
MAE first averages repeated observations per test solute within each coordinate.
A training-solute-balanced constant has macro MAE 0.890. A improves that reference,
but B/C do not improve A. These are not established HSP comparators.

Nominal 90% intervals use calibration-family maximum absolute errors. Mean complete-
family coverage is 0.939/0.891/0.905 for A/B/C; minimum coordinate coverage is
0.800/0.625/0.500. Mean full width is 4.927/4.983/6.077 log units. Broad intervals
and mean coverage establish neither precision, every-coordinate calibration nor
a simultaneous guarantee. Across coordinate-specific test sets, 1,190 rows exceed
training chemical-feature or temperature bounds; unseen families are still OOD
inside those scalar bounds.

Independent verification restores all **66 saved JSON heads** and reproduces every
calibration/test prediction, metric, solute-balanced score, calibration radius,
row/family coverage, macro summary and constant baseline within 1e-9. Full reserved
and partition aliases are also checked. Initial extraction failed before encoder
execution because the heavy-atom-count column was missing; the preserved failure
receipt and corrected cache record the exact existing 105-column recipe.

Prepared array SHA-256:
`471b35feb654e89fa2efd136192b2d0d69603a2b81af978242c40dfa64f04345`.
Feature cache SHA-256:
`0746fc2be68027b9639ea2e912e66a91067c51ce6f087af4e96ab4085fac152b`.
Plan SHA-256:
`89fc0d410f93665b080d1017d5b0bbd349188a3d6480a726b43592d264729ed7`.

The [full HSP/material objective](local-research-protocol.md) remains unachieved.
Source logS, including solid-form and entropy effects, does not identify three
absolute cohesive-energy components. Any learned representation needs independent
HSP calibration and a no-auxiliary control before HSP transfer. Polymer/EO–PO
physical definitions, applicability and measured validation remain required.
