# Crystal16 source qualification

This is the 2026-10-13 local intake and metadata-family qualification snapshot.
No transmission fitting was performed.

The local intake of the [author's exact Crystal16 export](https://github.com/mona2442/Polymer-Solubility-Data_Crystal16/tree/169a11bfd10886ee1577254b09236a5f244af2c2)
retained all 92,031 rows. These are processed optical transmission records, not
92,031 independent experiments, equilibrium solubilities or HSP3 measurements.
The [primary article](https://doi.org/10.1021/acs.jpcb.4c06500) reports 39,407
processed cooling points. That difference was unresolved at this intake. No model
was fitted in this operation, and it does not extend the default HSP applicability.

The CSV is 9,700,024 bytes, Git blob
`4d25165290436db1700bc071fa6d5d62a598ab41`, SHA256
`265bc28e3ff17f23ed150db37dfe0c714e5e0f6a5f73720af8fbbf9043916fe4`.
Its 12 original fields include temperature, transmission mean/median/standard
deviation, concentration/unit, polymer name, two source mass fields, solvent name
and two structure strings. It does not include reactor identity, cycle, heating,
cooling, holding or filtering state. Original strings, row indices, duplicates,
disagreements and missing values are retained in ignored local files.

The intake found 30 polymer names, 39 name/mass/structure tuples, 29 raw polymer
structure strings and 45 solvent names/strings. There are 824 extra exact duplicate
rows. At the same lexical metadata key, 16,862 groups contain differing reported
transmission values, covering 59,932 rows. Those differences are preserved;
they have not been resolved by averaging, voting or label-based filtering.
Both source mass fields are missing or invalid in 9,128 rows. Lexical counts do
not establish chemical identity, grade equivalence, independent replicates or
family isolation.

The author README describes cooling data. Static inspection of the pinned
processing notebook found a SQL query for filtered cooling records, source
uncertainty exclusions and specific program conditions. The query retains reactor
and state fields omitted from the CSV. The notebook has no CSV export statement
that binds that query result to this exact file. Its subsequent name-based
structure joins could also expand rows when a name has multiple structures;
this possibility has not been established as the cause of the count difference.
The author code was inspected as text; it was neither installed nor executed.
Notebook output values were skipped. The current local target description is
`experimental_transmission_processed_author_export`, with the author's cooling
claim recorded separately and the exact processing lineage unqualified.

The initial source-role proposal incorrectly called SI002 a polymer structure
file. Acquired bytes and separately reviewed cells establish these actual roles:

| Publisher SI asset | Observed content | Source SHA256 |
| --- | --- | --- |
| SI001 PDF, file 51132745 | Material/solvent definitions; first four pages inspected, Table S1 visually checked | `44582c6bae33cd697bc66ed9ccdbfea35691758c762a710060ca5187d04694d8` |
| SI002 XLSX, file 51132748 | `solvents_smiles`, 48 rows × 3 columns, including the header | `7059d633fadcfa8b605e5ef33ee1b66ad906745fc91895f742f5c3b0911f466d` |
| SI003 XLSX, file 51132751 | `polymers_smiles`, 33 rows × 4 columns, including the header | `1e116bb8618f4d29dc9925e32b6a2f26e396677e577f44a95b97d273c0ab8adb` |

The old role proposal and receipts remain intact, with an explicit correction.
SI003's extra acronym column caused the first structure-only header guard to
refuse; a distinct header review and corrected read followed. No source workbook
was modified. Table S1 distinguishes Mw minimum/maximum from Mn and PDI. The
CSV's `number_average_mw_*` names do not resolve that semantic conflict. Missing
chain mass, ends, DP, tacticity and distributions remain unknown. A repeat-unit
mass is not a chain mass. PEG and PEO grades must stay within their polymer family.

The GitHub dataset has no assigned dataset license. The article's CC BY license
does not clear those source data. The three SI assets are separately CC BY-NC 4.0.
The local operation records `audit_basis="operator_assumption"` and
`use_scope="scientific_private_noncommercial"`; it grants no public redistribution
of source observations, coefficients, source files or trained weights. This page
contains source metadata only. No numerical observations or source assets are
bundled with the MIT code.

The separate metadata-only family operation retained the nine original metadata
fields and all 92,031 row identities without parsing optical targets. It preserved
all 5,176 existing HSP reservation aliases and the current molecular Murcko/core,
complete bounded tautomer and periodic-repeat links. Source names and grades also
link into whole material components. The resulting 39 grades form seven
conservative components, with **zero eligible rows and zero eligible families**.
Thirty-eight grades have reservation hits; unresolved member identities propagate
through components covering 30 grades. Thirty-six grades were individually
structure-qualified, but that does not override component-level exclusions. Three
grades lacked an exact polymer-SI graph match and one had incomplete molecular
aliases. These reason counts need not be disjoint.

The metadata result prevented fitting. It supplies no evidence of transmission
prediction quality or HSP accuracy. The original structures remain the prospective
feature inputs; grouping aliases do not replace them. Unknown mass semantics keep
the MW feature disabled. Solvent aliases/reservations were recorded as metadata
for fixed-category candidates, with solvent-structure fitting still disabled.

[Issue 88](https://github.com/myu65/hansenkit/issues/88) tracks remaining source
lineage, grade/condition correspondence and independently reproducible family
qualification. Repeated conditions must stay within whole material families.
Any future transmission task remains separate from HSP training and accuracy
reports. HSPiT Excel remains evaluation-only: it cannot supply training or auxiliary
targets, fitted features, model selection or calibration. Reservations were not
weakened to make this intake trainable.
