# Rights and source ledger

Reviewed 2026-10-10. A publisher declaration and a legal clearance decision are different records.
This repository distributes its original MIT source only, with dependency references in `uv.lock`.
It does not vendor dependency wheels, external weights, HSP numerical tables or external teacher data.

| Asset | Source / declared license | Initial decision | Remaining work |
| --- | --- | --- | --- |
| Project code, SMARTS rules, schema, synthetic generator | Original work; [MIT](../LICENSE) | Included | Retain MIT notice |
| Synthetic label formula | `hansenkit.data`; original arbitrary formula, MIT | Included as generator; no physical coefficients | Never call its results measured HSP |
| RDKit | [Official license](https://github.com/rdkit/rdkit/blob/master/license.txt), BSD-3-Clause | Default dependency | Preserve upstream notices if packaging binaries |
| NumPy | [Official license](https://github.com/numpy/numpy/blob/main/LICENSE.txt), BSD-3-Clause plus bundled-component notices | Default dependency | Audit the exact wheel if redistributing it |
| scikit-learn | [Official license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING), BSD-3-Clause | Default dependency | Same; review transitive packages |
| Pydantic | [Official license](https://github.com/pydantic/pydantic/blob/main/LICENSE), MIT | Default dependency | Review core and wheel notices |
| LightGBM | [Official license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE), MIT | Optional dependency | No bundled native binary here |
| Hatchling build backend | [Official license](https://github.com/pypa/hatch/blob/master/LICENSE.txt), MIT; build-system requirement in `pyproject.toml` | Isolated build dependency, not vendored | Its build-isolation dependencies are outside the installed-runtime inventory; audit/pin the full build environment before release |
| Transitive and development packages | Installed metadata and license-file hashes in [dependency inventory](dependency-inventory.json); exact downloads/hashes in `uv.lock` | Declarations recorded; not vendored | Keep inventory in sync with lock/environment; binary redistribution needs a separate audit |
| IBM MoLFormer code | [Official license](https://github.com/IBM/molformer/blob/3b9ac434db387fadf2cf99b99def654cbf193841/LICENSE), Apache-2.0 | Not imported/copied | Exact adapter code, notices and full dependency review: [#3](https://github.com/myu65/hansenkit/issues/3) |
| MoLFormer-XL-both-10pct weights | [Publisher model card](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct), declares Apache-2.0 | **Disabled; not downloaded** | Verify exact revision, actual weight terms, hashes, custom code, dependency versions and training-corpus provenance before use |
| Full MoLFormer-XL checkpoint | [Publisher README](https://github.com/IBM/molformer) distinguishes offered 10% checkpoint from full XL | Not used | Do not describe the 10% model as full XL |
| HSPiT coefficient/Excel tables | [Upstream repository](https://github.com/Emil-Kongsbach/HSPiT) has MIT code; `src/HSPiT/stefanis_data2.xlsx`, `fitting_data.xlsx`, `hoftyzer_data.xlsx`, UNIFAC tables | **Blocked, not read/downloaded** | Asset-specific provenance/permission, underlying publications and source tables: [#2](https://github.com/myu65/hansenkit/issues/2) |
| HSP-predictions HSP_SMILES.csv / teacher weights | [Upstream repository](https://github.com/darjacvetkovic/HSP-predictions) has MIT code and research data/models | **Blocked, not read/downloaded** | Dataset and weight rights separately; original HSP source and any commercial database restrictions |
| Cleared user-owned CSV or fixed vectors | Operator-supplied local files + checksum-bound manifest | Local only after declared approval; not included here | Human review of evidence, permitted uses, confidentiality and release rights |
| Uni-Mol2, MiniMol, CheMeleon | No asset/version selected | Future; disabled/unimplemented | Code/weights/data/dependencies separately reviewed in [#6](https://github.com/myu65/hansenkit/issues/6) |
| QM9 auxiliary data | No data selected | Future; not downloaded | Dataset-specific terms, source lineage, units and auxiliary task boundaries |
| OpenMM/RadonPy and force fields | No dependency installed | Future interfaces only | Code, parameter and force-field rights; physical validation: [#7](https://github.com/myu65/hansenkit/issues/7) |

No external teacher was run and no external pseudo-label was produced. CI creates only original
synthetic labels and does not upload datasets or model artifacts. MIT on our code does not relabel
any external software, weight, table, or dataset as MIT.

Risk controls: deny-by-default manifests, separate label kinds, ignored local directories,
no download API, no private artifacts in CI, review before any public trained-weight release.
These controls cannot prove the truth of a user's rights declaration or detect a renamed copied dataset.
Retain permission evidence outside the repository if it is confidential; keep only permitted references public.
