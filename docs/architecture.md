# Minimal architecture

The package is a local research CLI and a few Python modules. There is no service, remote data
ingestion, automatic model download, model registry service or GPU dependency.

`chemistry → schema/scope → manifest-checked CSV → grouped split → train/calibrate/evaluate → JSON model → guarded CSV prediction`

## Chemistry and schema

RDKit sanitizes SMILES, removes removable explicit H and atom maps, and produces canonical
isomeric SMILES. Charge, isotopes, stereochemistry and disconnected fragments are retained.
It does not normalize tautomers, neutralize salts or take a largest fragment silently.

Twenty original SMARTS rules assign nonoverlapping heavy-atom ownership in priority order.
Group counts and seven RDKit descriptors form 27 chemical features. Ester/amide context atoms
are not counted as owned atoms. Unknown atoms stay unassigned; no catch-all coverage filler
pretends to support them. Coverage is a vocabulary check, not validation of a chemical theory.
Implicit hydrogens influence RDKit donor/acceptor/descriptors; heavy-atom coverage does not claim
that hydrogens have independently fitted HSP group contributions.

Pydantic inputs cover a molecule, polymer, surfactant and mixture. Polymer repeat units require
two singly attached dummy ports and fractions summing to one. Mn is in g/mol, dispersity ≥1;
end groups each have one port. EO/PO distributions preserve means, optional standard deviations,
sequence type and an empirical joint count PMF with probability/mean validation.
The [bounded chain implementation](chain-features.md) now assembles oriented ports, validates
mass/Mn and explicit EO/PO blocks, and checks intensive features against longer direct chains.
Unknown ends/tacticity/distributions remain explicit limitations. No validated polymer HSP
property backend follows from these structure computations.

Initial prediction scope is conservatively screened neutral connected organic small molecules
at 298.15 K with complete feature coverage (supported elements C/N/O/S/F/Cl/Br/I, MW ≤500,
≤60 heavy atoms). Repeated EO/PO ether chains, charged atoms including zwitterions, radicals,
dummy atoms, mixtures and structured polymer/surfactant inputs are refused. This is a PoC
eligibility screen, not experimentally validated HSP applicability.

## Three model paths

| Path | Runnable default | Planned scientific comparison |
| --- | --- | --- |
| A | Standardized chemical features + multi-target ridge; optional LightGBM | Chemical contribution baseline on independent measurements |
| B | Fixed 256-bit radius-2 Morgan fingerprint + ridge (**B-proxy**) | Licensed frozen MoLFormer vectors + regression |
| C | Linear chemistry fit + group-cross-fitted fingerprint residual ridge (**C-proxy**) | Linear chemistry + licensed frozen embedding residual |

C residual labels are computed from inner group folds of the training split; its final chemistry
head is refit on all training rows. Scaling is fitted inside each fold. Calibration and test groups
never train a head or residual. Hyperparameters are fixed for this initial plumbing experiment.

`Encoder.transform` and `metadata` support Morgan and approved local frozen-vector tables. The
table carries weights/code licenses, dependency review, exact revision, hash and permitted uses.
Model artifacts retain encoder identity and must reload with identical metadata. Missing vectors
fail closed. `--encoder molformer` requires the optional extra and explicit local checkpoint/
review paths. The audited adapter produces fixed 768-dimensional vectors offline with exact
tensor parity checks. `checkpoint-fetch` is an opt-in download; defaults still have no torch/
transformers dependency. See [the pinned audit](molformer-review.md).
Vector extraction, fit-on-training-only transformations of vectors and audit of any training data
used by a user-supplied encoder are the operator's responsibility.

`PhysicsBackend.simulate/provenance` is a future interface only. Local QM9 auxiliary regression
is implemented separately from HSP labels. [SolQuest preparation](solvation-auxiliary.md) adds
an optional streaming contract for computed interaction targets; its exported data still needs
fitting clearance. Alternative encoders, GPU MD and active learning remain roadmap work.

## Artifacts and scope

Models are plain JSON with numeric ridge parameters or LightGBM model strings. They contain
training structures, source manifest, versions, feature bounds and calibration radii, so even model
files can reveal private data. Keep them ignored/local. Loading does not use pickle or execute
remote code. Exact RDKit/NumPy/scikit-learn versions are checked for reproducibility.

The public prediction route checks scope, training-only Morgan nearest-neighbor similarity and
chemical feature-range extrapolation before returning numbers. OOD/unsupported rows contain
status/reasons and blank numeric CSV fields. Raw regression scores on held-out OOD molecules
are still needed inside evaluation to measure failure, but do not become public inference results.
All outputs include label kind, units and `real_accuracy_validated=false` in this unvalidated PoC.
