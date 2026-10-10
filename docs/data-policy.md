# Data and model rights policy

The initial demonstration uses structures authored for this project and arbitrary synthetic labels.
Its generator is original MIT code; its values are not empirical HSP or literature coefficients.
No third-party numerical table, training dataset or pretrained weight is included.

All external assets require separate review of source, exact version/checksum, rights holder,
license/evidence, permitted local evaluation, training, redistribution, derived labels/weights,
commercial use and attribution. Unknown is not approved. A manifest is an operator declaration,
not proof of ownership; maintainers must inspect the evidence before release.

HSPiT `stefanis_data2.xlsx` / `fitting_data.xlsx` and HSP-predictions `data/HSP_SMILES.csv`
are blocked pending asset-specific permission. MIT code licenses do not establish dataset clearance.
Do not train on them, distribute them, use them as coefficients or generate public pseudo-labels.
Explicitly requested local structural inspection/classification may read headers and reserve
molecular identities; keep workbooks outside the public repository. This is not numerical dataset
clearance. Formal local evaluation still requires asset-specific terms and measurement provenance.

User-supplied cleared CSV data can be read locally with an explicit manifest and
content hash. Private/company data and local model artifacts stay ignored. No upload command
is supplied. Public trained weights are a separate release decision and need manifest review.

Synthetic pipeline tests, teacher reproduction tests and independent experimental tests must
have separate label kinds and reports. Teacher agreement establishes only imitation of that
teacher. Experimental truth must be independent of both training labels and the teacher.

For the user's 2026-10-10 local research run, audits are assumed passed. Record the session
authorization and exact asset hashes in local manifests with `audit_basis="operator_assumption"`.
This scoped assumption supersedes the pending local-use gate above for that run; it cannot
authorize redistribution. HSPiT workbook numbers remain evaluation-only, including their
coefficient tables: no pseudo-label generation or training-feature construction from them.
Published compilations of unknown experimental/estimated lineage use `published_reference`.
An auxiliary QM9 row has quantum-property targets, not HSP targets; count those separately.

The rights ledger and follow-up review issues are available in the repository. Evaluation-only
manifests cannot authorize training even when other permission flags are mistakenly enabled.
