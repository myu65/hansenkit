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

The official PMC S3 release supplied both article and supporting PDFs, checked
against the per-object publisher MD5. Read-only transcription retains 381 numeric
polyester/solvent scores and 273 ASD/API/solvent scores, with 249 and 195 missing
cells respectively. The 2026 per-column totals reconcile exactly. These 654 pairs
are empirical compatibility scores, not 654 independent HSP compounds. Keep
source header/caption inconsistencies and unspecified solvent isomers unresolved.

A fixed-head diagnostic compares 15 unambiguous common solvents for each of the
three APIs, using no fitted threshold or additional training. A/B/C Spearman rho
is 0.660/0.757/0.725 for carbamazepine, 0.803/0.607/0.700 for griseofulvin, and
0.775/0.775/0.760 for resveratrol. These are explored one-minute, room-temperature
dissolution scores, with solid-form and numerical temperature uncertainty; calculated
solvation targets are at 300 K. Griseofulvin is outside the chemical training feature
box. No HSP predictions or equilibrium-solubility accuracy claims follow. The
diagnostic uses historical auxiliary heads; the current [stronger isolation audit](tautomer-isolation.md)
requires their refit before any qualified transfer claim.

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

The [primary PEESE property repository](https://github.com/PEESEgroup/Pure-Component-Property-Estimation)
at `9b386f845071596e926e375d95666f7a8b1e0ec6` explicitly names software and data
under MIT. Local checksummed identity-column reads find 1,037/1,017/1,016 D/P/H
component rows. The numerical `Experimental` header does not certify measured
lineage. Against the prior strict reservations, only three potential identities
have all components outside the known families; one is a reserved compound's
tautomer, and the source's nicotine graph needs independent positional-identity
review. No coefficients, predictions, feature-count table or HSP labels supply
training. This source has not yet yielded a qualified calibration population.

## Measured solubility sources, separate from HSP labels

The [EnSol primary paper](https://arxiv.org/abs/2609.21151) links the
[authors' repository](https://github.com/thaonguyen217/EnSol). At
`946a2addbd98b9f30f02bf494a0059c2e5c284e5`, static CSV downloads pass their Git
blob hashes: BigSol has 100,570 records for 1,375 raw solute strings, SolProp has
6,236 records, Leeds has 1,469, and the lab assay has 100 pairs for ten solutes.
No upstream code, pickle or weight is executed/downloaded. No repository license
is detected; code/weights remain disabled and dataset/source rights are separate.
The lab CSV retains 22 explicitly censored concentrations with nonfinite logS;
its temperature field is 273.15 without declared units. Keep that source field
and unresolved method conditions; do not infer or replace a temperature.

The [author-hosted BigSolDB v2.1](https://huggingface.co/datasets/levakrasnov/BigSolDBv2.1)
declares CC BY 4.0. At `8d7442c123d417fe4c18fd91524c176ede359f27`, the CSV passes
publisher LFS SHA-256 `212cb511e05051917f7ab8811282c06c99512e4f720cf270be555fdf07b4a854`.
It contains 112,465 solubility rows, 1,525 raw solute structures and 214 raw solvent
structures (218 solvent names in the card). Its `Source` column supplies literature
DOIs, and temperature is explicitly in K. There are 3,187 nonfinite logS cells and
3,925 repeated canonical solute/solvent/temperature rows. Preserve missingness,
source conflicts and original density-based concentration conversions; no clipping
or automatically averaged duplicate measurements have been adopted.

Current HSP reservation and conservative neutral-organic qualification leave 563
candidate canonical solutes and 38,294 associated rows in v2.1. They are not a
cleared fitting population: name/CAS/stereo/phase/method reconciliation remains
incomplete. Every complete pair includes a reserved or unsupported solvent;
zero pairs pass qualification for a **trainable molecular pair model**. A future
fixed-solvent-coordinate task needs its own reviewed contract and prospective
solute-family partitions; the solute-only count cannot silently authorize solvent
feature fitting. No new HSP triples or public model follow from this inventory.
Repeated temperature/solvent observations are not distinct compounds. A subsequently
qualified, separately reviewed [fixed-coordinate reference logS experiment](fixed-coordinate-solubility.md)
uses independently corroborated solute graphs and performs no solvent-graph fitting.
Its preparation, support rules and results are separate from the original
permission-off inventory and make no HSP accuracy claim.

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
