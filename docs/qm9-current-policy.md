# Current-policy QM9 audit and refit

The original characterized QM9 population contains 88,599 rows. Its original
normalization and canonical-then-strict-Murcko union order were reconstructed,
yielding exactly 69,142 train / 19,457 test rows and 12,478 / 3,120 groups. Reloading
the old head and matching the original frozen embeddings reproduces all six saved
MAE/RMSE values. This verifies the audited population without replacing the original
split with today's coarser grouping or modifying old source/model artifacts.

Under `canonical+strict-murcko+core-topology+bounded-tautomer-v1`, 10,006 historical
training rows and 2,639 test rows intersect HSP/Excel reservations. 2,570 aliases
cross the old train/test split, involving 60,092 rows. These counts overlap and
must not be summed. The historical head is not a current-policy isolated model;
filtering test alone would not repair its fitted training population.

Structure-only preparation excludes 12,645 reserved rows, retaining 75,954 rows
in 6,651 connected groups. All retained enumerations complete. Targets, original
SMILES/encoder inputs and fixed embeddings remain unchanged. The new immutable
plan uses seed 42, Ridge alpha 100 and the reviewed frozen MoLFormer representation:
61,622 training / 14,332 test molecules, in 5,320 / 1,331 groups. No HSP values or
Excel coefficients enter selection, fitting, features, calibration or pseudo-labels.
A separate operator-assumption fitting review binds the new selection and plan;
source preparation itself is not fitting permission or redistribution permission.

| Quantum target | MAE | RMSE | R2 | Unit |
| --- | ---: | ---: | ---: | --- |
| mu | 0.654 | 0.865 | 0.591 | Debye |
| alpha | 2.122 | 2.781 | 0.858 | a0^3 |
| gap | 0.01451 | 0.01879 | 0.837 | Hartree |
| cv | 0.923 | 1.199 | 0.885 | cal/mol/K |
| u0 | 9.990 | 13.202 | 0.867 | Hartree |
| r2, electronic spatial extent | 67.513 | 88.313 | 0.747 | a0^2 |

The `r2` target is electronic spatial extent; the statistical R2 column is a
different quantity. In particular, absolute `u0` errors of about ten Hartree are
large: a positive statistical R2 is not accurate thermochemistry or cohesive-energy
validation. No condensed-phase HSP or large-molecule transfer claim follows.
Historical targets had already been explored; these results are not a new blinded
benchmark, and changed populations prevent attributing metric changes to a better
estimator. No test-based method selection or refitting is performed.

A separate process checks all selected identities against the original embedding
rows, all cached aliases against current HSP reservations, and all train/test alias
links. It restores the saved JSON head and reproduces every test prediction and all
six MAE/RMSE/R2 metrics within 1e-9. These are quantum auxiliary observations,
**zero HSP training rows**. External arrays, fitted states and values stay local.

Public `auxiliary-fit` also records `grouping_policy` in both its report and saved
state. A synthetic fixture checks that saving JSON preserves that provenance.
Old metadata and metric reports remain historical; they are not silently upgraded.

Selection SHA-256:
`6a9bcce96439bc0c9a4ad5c2e2cb4dc4e8f548159fb4f68c0a54892aba12f624`.
Plan SHA-256:
`5b848cb8bb8268658ff817d287866de688e690d345e7dc83f2b47f89eb80ec9c`.
Fitted head SHA-256:
`b88480d00a1e808fb76750f17946b7be7c06a0f5f712f78940aa3f679a360194`.

Next, test complementary chemical features and transfer against a no-auxiliary
control after independent HSP calibration labels exist. Keep the full
[HSP objective](local-research-protocol.md) and [material validation](chain-features.md)
open; quantum training counts and finite structure arithmetic cannot satisfy them.
