# Reference identity audit and expanded evaluation reservations

Local review, 2026-10-10. External structures, responses, tables and fitted heads remain
outside Git. This report records aggregate evidence, not a cleared experimental dataset.

## Complete CAS batch, incomplete scientific reconciliation

[PubChem Identifier Exchange](https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange-help.html)
returned input/output correspondence for all 1,210 requested identifiers from the reference
CSV, Mathieu supplement and unresolved material components. All candidates are retained.
The operation was Same CID with SMILES output; no parent, component, stereo-removing or
connectivity-only transformation was requested. No numerical HSP values were submitted.

The 1,197 identifiers with RDKit-valid source structures include seven identifiers with
multiple distinct candidates. Forty-two earlier PUG REST response candidate sets agree
with the batch. These are source consistency checks, not experimental HSP verification.

| Classification of the 1,192 source CSV rows | Rows |
| --- | ---: |
| Exact canonical structure agreement | 1,108 |
| Stereo specificity difference | 29 |
| Molecular formula difference | 9 |
| Constitutional structure difference | 8 |
| Multiple source identities; review required | 4 |
| No source structure available | 33 |
| Tautomer difference | 1 |

Twenty-five CSV identifiers fail the single-CAS format/check-digit check. This includes
unusable identifiers; do not assume each represents a one-digit typo. The
[CAS check-digit documentation](https://www.cas.org/training/documentation/chemical-substances/checkdig)
defines that check. No source row was automatically replaced, neutralized or desalted.
Names, salt composition, stereochemistry, ambiguous candidates and conflicting duplicate
labels still need review. [Issue #15](https://github.com/myu65/hansenkit/issues/15) remains open.

The exact matching rows describe 1,091 unique graphs; 1,034 are connected organic graphs
without formal atomic charges or radicals. The default PoC structure gate accepts 954.
These counts precede duplicate-label and reference-lineage adjudication. They do not prove
1,034 measured HSP triples, qualified numerical predictions or a new blinded benchmark.

Checksums:

| Local asset | SHA-256 |
| --- | --- |
| Original `HSP_SMILES.csv` | `8ec0f0a6012a339ea61778ad236be9d08080a3ed90f7b9d3f33b565d7f9d0ae3` |
| Complete PubChem input/SMILES correspondence | `9dd03243d9cf14a72e9cae83df4ccf0e0644e9bf26cfc94dde3ee239c25e6386` |

## Expanded Excel reservation and auxiliary rescore

The Mathieu supplementary spreadsheet remains evaluation-only as documented in
[the original source audit](mathieu-reference-audit.md). Independently returned graphs,
linked original graphs and material components are conservatively reserved without
rewriting prediction inputs. The known Excel reservation now contains 975 identities and
111 Murcko families. Only 19 of the historical 1,030 reference structures remain eligible
for a future HSP fit under this reservation. Unresolved source identities still prevent
full reconciliation; do not proceed with a new fit by ignoring those identities.

Reserving all corrected HSP-reference candidates as well expands the auxiliary reservation
to 1,262 identities and 131 families. Reconstructing the saved QM9 split shows zero known
new-family conflicts among the 69,142 fitted rows. There are 317 conflicts in its previous
19,457-row test partition, all confined to two test families. The fixed saved head reproduces
its previous metrics; removing those test rows leaves 19,140 rows and 3,118 groups, with no
training/test family overlap. No head, preprocessing or encoder was refitted or selected.

| QM9 target | Filtered MAE | Filtered RMSE | R² | Unit |
| --- | ---: | ---: | ---: | --- |
| Dipole moment | 0.758391 | 1.009445 | 0.530162 | Debye |
| Polarizability | 2.259797 | 2.980907 | 0.839556 | a0³ |
| HOMO–LUMO gap | 0.014509 | 0.018472 | 0.849663 | Hartree |
| Heat capacity | 0.968095 | 1.256593 | 0.874391 | cal/mol/K |
| Internal energy U0 | 10.430560 | 13.769221 | 0.856146 | Hartree |
| Electronic spatial extent | 73.269297 | 96.474951 | 0.725218 | a0² |

These are auxiliary quantum-property metrics, not HSP metrics. Known-reservation checks
do not certify unresolved Excel identities. Historical results and their original smaller
reservation are retained; the expanded audit must accompany any future reuse.

## Additional related repositories

[KouTer](https://github.com/cassimon/kouter), revision
`6230b731ed42ccf6f984c1b874d65cfc723d3972`, has MIT code and a 4,962,580-byte prediction
pickle. Static opcode inspection, without unpickling or executing downloaded code, finds
HSP Value/Std_dev fields normalized by MAD and a serialized RangeIndex from 0 to 5,204.
This index is transport metadata, not an independently verified unique-compound count.
The [primary manuscript](https://arxiv.org/html/2606.13060v1) applies a QM9-pretrained
DimeNet/descriptor/GP pipeline to CompSol candidates; its HSP references come from a
handbook. The prediction table supplies no additional independently measured HSP labels.
Upstream-table rights, normalization factors, training overlap and calibration must be
checked before numerical comparison. Never use these predictions as Excel-independent
pseudo-labels. Local pickle SHA-256:
`cbefdff0d1546fe286d7a95f11fd9cdbd113f0e001f72b6a5ea345214d225369`.

[TLGNNPPSP](https://github.com/RuiyiFang/TLGNNPPSP), revision
`5f6c4076c183db42bfef84c3ec1181bbb33a90be`, describes solvent-to-polymer transfer and
plateau constraints. Its inspected tree and parent repository lack the solvent/polymer
CSVs and saved weights named in the README; no repository license was detected. Reported
README scores cannot substitute for a local reproduction or independent polymer validation.
No code, weights or data were imported into hansenkit. The architecture is a research lead;
its missing assets and HSPiP-named source tables need separate review.

## Next experiment

Resolve or explicitly adjudicate identity and duplicate-label discrepancies, reserve the
full evaluation population, and obtain additional measurements outside the reserved
families. Freeze candidates, calibration partitions and comparisons before fitting.
The current 19-row HSP training remainder cannot support a claim of established-method
accuracy for 1,000 compounds. More embeddings or predicted labels do not remove this
evidence gap. Polymer HSP and long-chain surfactant HSP remain unqualified.
