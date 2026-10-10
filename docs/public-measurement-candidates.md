# Public HSP measurement candidates

Read-only source qualification, 2026-10-10. The user has no private cleared HSP
dataset and requested a public-source search. Source publication records, unique
chemical identities, polymer grades and usable training examples are different counts.
No source numerical table is included in this repository.

## Experimental material studies

Two primary papers provide 28 candidate publication records: 25 polymer-grade or
composition records and three molecular APIs. These are not 28 certified distinct
compounds or an accepted training/evaluation set.

| Primary source | Observed records | Method and qualification |
| --- | --- | --- |
| [Patel et al., 2024](https://doi.org/10.1021/acssuschemeng.3c07284) | Table 2: 15 polyester grades/compositions | Film swelling/solubility over five days; normalized solvent uptake; HSPiP 5.4.01 genetic-algorithm sphere fit. Table 1 supplies molecular-weight and thermal metadata. Retain grades, composition and crystallinity. |
| [Experimentally Derived HSP, 2026](https://doi.org/10.1021/acsomega.6c05058) | Table 2: ten ASD excipients and three APIs | Polymer swelling over four days; empirical solubility scores and HSPiP sphere fitting. Composition/grade metadata in Table 1. Ionic, graft and variable-substitution materials require separate scopes. |

Both article XML releases explicitly declare CC BY 4.0. Supplementary assets and
third-party solvent-reference tables still need their own provenance review.
The measured observations are solubility/swelling; HSP triples are experimentally
derived sphere centers, with thresholds and the solvent-reference convention part
of the method. This is distinct from a structure-prediction teacher, and also from
a direct measurement of three separately resolved cohesive energies.

Local checks preserve one mismatch between a reported total parameter and the
Euclidean norm of its three components. The source values remain unchanged, and
the record is flagged for review. Numeric temperature is unset where not established
for the relevant experiment; room-temperature aging, NMR or a heated miscibility
test does not establish a 298.15 K HSP determination.

All 28 records remain **evaluation candidates only**: zero newly qualified training
rows, zero accepted independent HSP test rows, no numeric predictions. Local
checksummed XML and transcriptions remain outside Git. A polymer with unknown
composition, end groups or molecular weight must not become one guessed SMILES.
Existing default polymer/ionic/mixture refusals still apply.

## Compilations and solvent-design tools

The [Wolfram 211-solvent resource](https://datarepository.wolframcloud.com/resources/JoshuaSchrier_Hansen-Solubility-Parameters)
identifies its source as the Zeng/Polymer Handbook table and Abbott's calculation
workbook. It supplies identifiers and a 25 C context, but it is a recompilation,
not 211 new independent measurements; table rights and overlap remain separate.

The [2026 paints/coatings solvent guide](https://doi.org/10.1039/d6su00271d) adds
sustainability scoring and solvent-mixture selection. Its methods source HSP from
existing Abbott/Accudyne lists. A local read-only inspection of the publisher's
`Sustainable_Solvents2.zip` finds 104 consumer and 106 industrial records, with 106
unique names in their union. The HSDX records lack CAS/SMILES and per-record
measurement-lineage fields. The article license does not by itself clear the
reused numerical tables. No source values, software or fitted model are imported.
The [authors' companion repository](https://github.com/helensneddon-arch/solvent-guide-hsp)
is a solvent-design source candidate, not evidence of new empirical HSP labels.

The [SolvPred authors' documentation](https://github.com/xueannafang/HSP_toolkit_docs)
describes inverse solvent selection and HSP location from experimental scores;
its solvent database cites PubChem and the HSP handbook. Such a tool can be a
method comparator after review, but its table is not automatically independent truth.

An identifier-only local inventory of the HSPiPy example's 1,206 rows leaves one
potential neutral organic structure outside the current reservations and identity
qualification exclusions. This does not clear its measurement lineage or permit
fitting. Recompiling overlapping tables cannot supply a new 1,000-compound benchmark.

## Next experiment

For [polymer validation #5](https://github.com/myu65/hansenkit/issues/5), first review
supporting-information solvent scores, test conditions and material specifications
from the two experimental papers. Freeze whole polymer/comonomer and commercial
grade families before fitting; reserve molecular API identities and scaffold families
against every previous fit. Keep room-temperature sphere estimates separate from
high-temperature melt-miscibility outcomes and teacher-model diagnostics.

For [small-molecule data #2](https://github.com/myu65/hansenkit/issues/2), continue
searching source-resolved measurements. A measured-label HSP head needs a cleared
calibration population outside the Excel identities/families; auxiliary computed
solvation targets cannot substitute for that calibration. Check the new SolQuest
[A/B/C results](solvation-auxiliary.md) before deciding which physical representation
to transfer. No test-error-based exclusions or split relaxation are permitted.
