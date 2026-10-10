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
The real EGP source passed its publisher MD5 and full-file preparation. Of 18,362 rows,
8,269 unique molecules and 322,491 computed target values remain: 6,943 rows are reserved,
1,995 charged/disconnected/radical/dummy, 1,140 missing/nonfinite and 15 nonorganic.
Per-row rejection reasons and original source indices reconcile the entire population.
The exported fitting permissions remain off because complete holdout reconciliation,
source method/outlier review and a prospective transfer experiment are still required.
No new SolQuest head or HSP head has been fitted. The larger GDB17 archive is downloading;
its data and preparation remain unverified. Track the ingestion/transfer in
[#20](https://github.com/myu65/hansenkit/issues/20). The complete 1,000-compound HSP
accuracy and polymer/surfactant property-validation objective remains open.

Local EGP SHA-256: `4d5f1de3fb7929ce9a7646be5d78faf683c61555d6ca615caca813a22fc7b701`;
prepared NPZ SHA-256: `7acf1678ea35581554c4768ad92f0868f31ed988c6f30c0dfde4c3a5cf16e687`.
