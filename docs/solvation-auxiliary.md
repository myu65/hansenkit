# Solvent-interaction auxiliary data

The [SolQuest primary manuscript](https://arxiv.org/html/2411.00994v1) and
[Zenodo release](https://zenodo.org/records/13952172) provide COSMO-RS solvation free
energies in 39 named solvents at 300 K. Release metadata declares CC BY 4.0. These
are computed interaction targets, not independently measured HSP triples. Dataset,
producer software, encoder and any derived model have separate provenance/rights.
The proposed transfer experiment retains the reviewed frozen MoLFormer encoder.

## Local preparation

```sh
uv sync --locked --extra solvation-data
uv run --extra solvation-data hansenkit solquest-prepare --source local/EGP.json --source-review local/solquest-source-review.json --reserved-identities local/full-hsp-excel-reservations.json --out runs/solquest-prepared
```

Supply an `AuxiliaryManifest` source review binding the **raw JSON or ZIP** SHA-256,
`label_kind="quantum_computed_auxiliary"`, all target names exactly as found under
`SOLVATION`, explicit `kcal/mol` units, rights/evidence and permitted local preparation
of data for training/derived weights. The nonempty reservations file contains a
`smiles` list for the full HSP evaluation/Excel population, optionally `series`.
This command performs no download, encoder execution, regression or calibration.

The optional pinned ijson parser reads only the `SMILES` and `SOLVATION` values into
memory; conformer geometries and large ECFP arrays are not materialized. ZIP input
must contain one JSON member (at most 64 GiB); no archive extraction occurs. The
flat EGP/GDB17 schema is supported; nested AMONS subsets need another reviewed
adapter and are refused by this contract.

Every target column must have exactly one scalar per SMILES, with the reviewed
column names. Negative energies are retained. Missing/nonfinite targets are rejected,
never replaced with zero. A bounded token reader handles the release's Python-style
bare NaN/Infinity/-Infinity as null/missing, preserving quoted and escaped text and
the original source file. Normalization retains charge, isotopes, stereo and fragments.
This auxiliary profile then excludes charged atoms, disconnected
graphs, radicals, dummy ports and nonorganic graphs. It reserves canonical identities
and the existing strict Murcko families, including the common acyclic family.
The profile does not broaden the default HSP inference scope.
Normalization must survive an unchanged RDKit parse/serialize round trip. An unstable
canonical identity is refused and tracked as `unstable_canonical_identity`; never
choose a stereochemical configuration by iterative serialization or lexical order.

Canonical duplicate vectors equal within 1e-8 kcal/mol collapse; all conflicting
duplicate identities are excluded. The report preserves original indices and reasons.
Rows, unique molecules and individual target values are distinct counts. Selecting
input records by HSP prediction errors is prohibited.

The output NPZ contains `smiles`, `targets` and `target_names`. Its checksum-bound
manifest deliberately has `training_allowed=false`, `derived_weights_allowed=false`
and `redistribution_allowed=false`. Resolve the complete holdout inventory, inspect
source method/outliers and preregister an experiment before issuing fitting permissions.
Filtering known identities alone does not prove that unresolved Excel compounds were
reserved. The source solvent contexts are named target coordinates; their HSP values
are never training inputs. The 300 K auxiliary context is distinct from the PoC's
298.15 K HSP context and must remain disclosed during transfer.

## Companion thermodynamic source inventory

The checksum-verified [CompSol corrected 2024 workbook](https://aip.figshare.com/articles/dataset/26871184)
also declares CC BY 4.0. It contains 38,513 pure-component temperature records for
1,834 identifiers, and 69,302 binary-system records. Only 29,009 binary records have
finite temperature/G/S/H values together. Pure-component H equals G+TS in the supplied
values, so these three columns must not be counted as three independent observations.
Temperatures include 298 K; do not relabel them as 298.15 K or assume room-temperature
liquid applicability. Source reaction warnings, pressure, missing data, provenance and
identities require review before a pair-task fit. The workbook adds zero HSP-label rows.
It was read locally without edits; no workbook, source values or derived heads are bundled.
After a complete additional 1,303-identifier batch and the known full HSP/Excel family
reservation, only 106 distinct pure-component structures (2,376 temperature rows) remain
candidates. Four ambiguous identifiers and 20 identifiers lacking independent structures
remain outside that candidate set. Name/structure and measurement-lineage review is incomplete.

## Experiment status

The preparation contract is tested on original synthetic interaction arrays in CI.
The real EGP source passed its publisher MD5 and full-file preparation. After the
stable-identity guard, 8,268 unique molecules and 322,452 computed target values remain
from 18,362 rows: 6,943 reserved, 1,995 charged/disconnected/radical/dummy, 1,140
missing/nonfinite, 15 nonorganic and one unstable canonical identity.
Per-row rejection reasons and original source indices reconcile the entire population.
The prospective molecular inventory classifies all 970 HSPiT/Mathieu source rows:
958 have all available candidates reserved, and 12 are excluded from single-molecule
numeric evaluation because of undefined material specifications (10), an invalid
identifier (one), or an isomeric mixture (one). Exclusions use identifiers/specifications,
never HSP values or prediction errors. All identified components remain reserved.
This adjudication does not certify unique structures, measurement lineage or a complete
material benchmark; those sources remain evaluation-only. The enlarged known reservation
contains 1,266 identities and 132 strict Murcko families.

Exported preparation-manifest fitting permissions remain off. The completed local
experiments use separate checksum-bound fitting reviews with
`audit_basis="operator_assumption"`, the qualified molecular evaluation inventory,
source method/outlier decisions and immutable prospective plans. They do not claim
that every excluded material has a unique resolved structure, or authorize any
public data/model redistribution. No HSP head has been fitted from these targets.

The complete GDB17 archive passed publisher MD5. Stable-identity preparation retains
189,764 unique molecules from 309,468 source rows, with 7,400,796 target values.
Rejections reconcile the full source: 40,055 reserved, 7,777 charged/disconnected/
radical/dummy, 71,855 missing/nonfinite, five invalid, six unstable and six conflicting
duplicate rows. The conflicting duplicate rows represent three excluded identities.
There is no source-target clipping, zero filling or HSP-error-based selection.

## Current-policy local A/B/C refits

After the complete historical audit, structures were selected again with
`canonical+strict-murcko+core-topology+bounded-tautomer-v1`. Eligibility and grouping
use no HSP values or source prediction errors. Fixed features/embeddings and source
targets are unchanged; the source targets remain computed solvation energies at
300 K. Ridge alpha 100 and the four-fold residual procedure are unchanged.
The new partitions were frozen before fitting; historical source results had already
been explored, so these are not newly blinded benchmarks.

| Experiment | Train / calibration / test molecules | A mean MAE / RMSE | B mean MAE / RMSE | C mean MAE / RMSE |
| --- | --- | --- | --- | --- |
| EGP, current-policy groups | 3,462 / 496 / 1,586 | 2.641 / 3.590 | 2.862 / 3.765 | 2.862 / 3.822 |
| GDB17, current-policy groups; all EGP cores reserved | 97,619 / 30,665 / 31,094 | 1.060 / 1.480 | 1.059 / 1.462 | 0.938 / 1.338 |

Units are kcal/mol. MAE averages all 39 target columns; RMSE averages the 39
per-target RMSEs. EGP has 408/137/137 train/calibration/test groups; GDB17 has
46,414/15,472/15,472. See the [selection and rejection counts](tautomer-isolation.md).
The full cached alias population has zero overlap with HSP reservations and zero
train/calibration/test alias overlap. An independent pre-fit check also verifies
every selected GDB17 structure against all 730 external EGP core keys.

C reduces GDB17 MAE by 11.5% versus A; B differs from A by only about 0.001 kcal/mol.
GDB17 mean per-target R2 is 0.704/0.708/0.757 for A/B/C. On EGP, B/C worsen MAE
and their mean R2 is 0.008/-0.023, versus A's 0.098. No model is selected or refitted
using these test results. Changed populations and partitions prevent interpreting
differences from historical metrics as an improvement caused by the isolation fix.

Saved JSON heads reproduce predictions; an independent process recomputes all 39
MAE/RMSE/R2 values, calibration-family-max radii and row/family coverage. GDB17's
mean per-target complete-family coverage is 0.900/0.899/0.900 for the nominal 90%
intervals; EGP's is 0.876/0.905/0.902. The minimum across EGP solvents is
0.839/0.869/0.861, so nominal coverage is not established across every target or
source. These are diagnostics, without a physical or exchangeability guarantee.
Feature-box OOD identifies 56 EGP test rows and three GDB17 test rows; all test
families are unseen even if their feature coordinates lie inside that box.

No new current-policy cross-source transfer result is claimed here. The historical
negative-R2 transfer results below remain a warning requiring a fresh fixed-head
diagnostic. No HSP targets, Excel coefficients or public trained weights result
from these fits, and no polymer/surfactant numerical backend is qualified.

Immutable EGP/GDB17 plan SHA-256 values:
`2a6bf9682b29410e131f684f982a8284a3040ddeb9ee69ed637eb58846eceb27` /
`57cb4991d778a03236ca5e2dc41feeac357e90353c4c79173dc3e42b31feb821`.
Selection SHA-256 values:
`e24b07c0e0db1c92d11f4c20e343d8d07a913e130784db652e2ffe6776688c77` /
`eb31710bfd7b4dacd51fa461f0ebad873bd5c9d278303d334800e2ae3fb8398f`.

## Historical local A/B/C comparisons

These are historical results under their original strict-Murcko policy.
The [stronger tautomer/core audit](tautomer-isolation.md) finds EGP and GDB17 training
and partition conflicts. Both populations have been selected again for new plans
and refits. The metrics below remain unchanged; they do not certify isolation
under the current grouping policy or qualify an HSP transfer backend.

All models use Ridge alpha 100; A has 105 original chemical features, B has the
reviewed frozen 768-dimensional MoLFormer representation, and C adds an embedding
residual head trained on four inner folds with whole training families separated.
The chemical features add absolute counts for extensive solvation targets to the
existing density descriptors. Incomplete group coverage is explicit; no catch-all
group is invented. Preprocessing, residual training and feature-box OOD bounds use
training only. Calibration/test populations and identities/families were disjoint
under the original strict-Murcko policy; the new audit finds stronger-policy links.
Per-target 90% intervals use calibration-family maximum absolute errors. They are
diagnostics, not guaranteed physical coverage or a simultaneous 39-target interval.

| Experiment | Train / calibration / test molecules | A mean MAE / RMSE | B mean MAE / RMSE | C mean MAE / RMSE |
| --- | --- | --- | --- | --- |
| EGP, strict held-out families | 5,034 / 690 / 2,544 | 2.905 / 3.747 | 2.998 / 3.864 | 2.934 / 3.853 |
| GDB17, strict held-out families; all EGP families additionally reserved | 107,078 / 36,055 / 38,081 | 1.078 / 1.502 | 1.058 / 1.462 | 0.942 / 1.341 |

Units are kcal/mol; each MAE is the average over 39 target columns. RMSE is the mean
of the 39 separately computed target RMSEs. These are **computed solvation** results,
not HSP triples, measurements, or evidence of 1,000-compound HSP parity.

The GDB17 experiment excludes a further 8,550 rows to reserve all 8,268 EGP molecules'
families, leaving 181,214 molecules. Train/calibration/test have 72,048/24,016/24,017
strict Murcko families. C improves GDB17 test MAE by 12.6% versus A; EGP's own
held-family experiment demonstrates no improvement from B/C over A. No test-based
model selection or refitting is performed. All saved model JSONs reproduce predictions
after reload; every solvent's MAE/RMSE/R2 and interval coverage is retained locally.

Historical source-transfer diagnostics apply each GDB17 model to all 8,268 EGP
molecules with zero overlap under the original strict-Murcko policy. The stronger
audit requires a new fit before a current-policy claim. This explored EGP population is not a new
blind benchmark. A/B/C mean MAE is 3.207/3.291/3.137 and mean RMSE is
4.390/4.243/4.361 kcal/mol. **Every solvent's R2 is negative for all three models**;
mean R2 is -0.279/-0.195/-0.262. A small relative MAE reduction does not establish
successful source generalization. 3,831 EGP rows are outside the GDB17 training
chemical-feature box, compared with 11 within-source test rows. All test families
are unseen even when their scalar features fall inside that box.

The actual frozen GPU extraction covers 189,764 GDB17 molecules; a fresh reviewed
encoder instance independently reproduces 32 structure-only probes from each EGP
and GDB17 cache within 1e-4. This is a sampled cache check, not independent re-embedding
of the entire population. Source geometry/SMILES correspondence remains unverified,
and source computed outliers are retained and disclosed. No encoder was fine-tuned.

Track transfer in [#20](https://github.com/myu65/hansenkit/issues/20), and qualification
of [new public measurement candidates](public-measurement-candidates.md). The complete
1,000-compound HSP accuracy and polymer/surfactant property-validation objective
remains open. Computed-solvation training counts cannot satisfy that objective.

Local EGP SHA-256: `4d5f1de3fb7929ce9a7646be5d78faf683c61555d6ca615caca813a22fc7b701`;
prepared NPZ SHA-256: `7acf1678ea35581554c4768ad92f0868f31ed988c6f30c0dfde4c3a5cf16e687`.
That NPZ is the historical pre-guard preparation. The stable-identity preparation is
`d3a6474feff61fda9c239a0139bb8361e885316b54da1a3e8a6d97c794ac1b5f`.
GDB17 ZIP SHA-256: `79185e43c28dd01893ea646c380e3024f09633abd59d30d4c6f3a6f4cac51518`.
GDB17 prepared NPZ SHA-256: `341c37e37935db2dd0fd2e8b6534286b930299306344ec3d3c3e25ac001c4655`.
GDB17 experiment plan SHA-256: `9396758d6a6b9ce6331d9cf0cbd7b9c9ea0ee9a2db2abfe2b6bda97a3f0d0ad4`.
