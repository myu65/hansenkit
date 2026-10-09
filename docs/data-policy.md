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
Do not download them into the project, train on them, distribute them, use them as coefficients,
or generate public pseudo-labels from them. Local evaluation is also subject to the actual terms.

User-supplied cleared CSV data may eventually be read locally with an explicit manifest and
content hash. Private/company data and local model artifacts stay ignored. No upload command
is supplied. Public trained weights are a separate release decision and need manifest review.

Synthetic pipeline tests, teacher reproduction tests and independent experimental tests must
have separate label kinds and reports. Teacher agreement establishes only imitation of that
teacher. Experimental truth must be independent of both training labels and the teacher.

The rights ledger and follow-up review issues will be added in the PoC PR.
