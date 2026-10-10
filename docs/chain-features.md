# Chain structure computation

`chain-features` implements oriented chemical assembly and intensive structure descriptors.
It is part of the polymer/long-chain work, not completion of an HSP inference backend.
The ordinary `predict` command still requires its model's independently validated scope.
These reports have `hsp_predictions=null` and `physical_accuracy_validated=false`.

```sh
uv run hansenkit chain-features --input examples/polymer.json --output runs/polymer-structure.json
uv run hansenkit chain-features --input examples/surfactant-block.json --output runs/surfactant-structure.json
```

## Connectivity and mass

Repeat units have two single-bond dummy ports; end caps have one. Distinct isotope labels
`[1*]` and `[2*]` specify left-to-right orientation. Unlabeled ports use canonical atom order;
use isotope labels for asymmetric repeat/end combinations. Atom-map labels are removed by
SMILES normalization and do not specify persistent port orientation. RDKit `molzip` joins
the corresponding attachment atoms with valence checks, removes the dummy atoms and
preserves atom and double-bond stereochemistry. Molecule formula, cap effects and stereo
are tested against explicit expected structures. No arbitrary attachment site is guessed.

For a homopolymer, one and two capped repeats determine residue mass and end-mass offset.
The number-mean repeat count follows from Mn. Unknown ends receive explicitly labeled
hydrogen-capped representatives, with an issue flag; this is not an assertion about the material.
Fractional end populations and multi-unit copolymer architectures need another distribution
profile. Material phase, tacticity and dispersity are not determined by Mn alone.

Surfactant caps must supply explicit ports. EO uses `[1*]OCC[2*]`, and PO uses
`[1*]OCC(C)[2*]`; tail-to-head connectivity gives the intended alkoxy chain without a peroxide
link. The supplied PO convention and unspecified tacticity are recorded. Mixed blocks require
an explicit order. Random or underspecified sequences and ionic/counterion materials are
refused by this profile. No neutralization or counterion removal takes place.

## Long chains and distributions

An explicit molecule is limited to 256 units, checked **before** allocating a repeated sequence.
Large Mn/EO/PO counts use bounded representatives and validated additivity of local
descriptors. Homopolymer populations are checked at 4/16/64 units; EO/PO block populations
are checked independently at 4/16/32. The additive populations and heavy-atom counts yield
normalized group/element counts, square-root counts and per-atom descriptors for the mean
chain. An independent direct 128-unit PEG and 80+80-unit block chain match those features.
Very large chains do not go through the MoLFormer token limit or silently truncate SMILES.

RDKit's default substructure limit is 1,000 matches. It can affect both group coverage and
Crippen LogP/refractivity atom typing in explicit long chains, including hydrogens and symmetric
matches. The research descriptor reads the installed RDKit BSD-licensed `Crippen.txt` in
its original priority order, types every explicit atom with unlimited local-pattern matches,
and refuses incomplete typing. The table is not vendored. Reports record the RDKit version,
table SHA-256 and feature version `original-density-descriptors-v2-complete-crippen`.
Small-molecule parity, more than 1,000 matches and direct 128-unit PS/PMMA/PA6 are tested.
Earlier local development models used descriptor version v1; retain their original reports
and do not silently recompute inputs with v2.

An empirical joint PMF retains zero-EO/PO bins and correlations when averaging descriptor
populations. Declared means/standard deviations must agree with the PMF. Normalizing the
mean populations is a moment representation; it is not the expectation of every nonlinear
property. Means of a broad/unknown distribution do not determine zero-block probability or
expected HSP. The approximation is recorded. Mn inconsistent with the explicit counts/caps
is flagged. These issues must inform the future property scope and uncertainty profile.

The original functional-group vocabulary can remain incomplete for P/Si or other chemistry.
The research descriptor uses explicit element counts and reports group coverage/missing
atoms; it does not silently assign a fictitious functional group. Complete elemental features
do not establish thermodynamic applicability.

## Connection to the accuracy goal

The next property backend needs polymer/series validation and source-supported HSP values
or separately identified solubility/MD endpoints. A stable feature limit alone does not prove
that δD/δP/δH are correct. Keep the complete 1,000-compound accuracy and polymer/surfactant
objective active until the final model and scientific evidence meet it.
