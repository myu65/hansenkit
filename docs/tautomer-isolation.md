# Tautomer and scaffold isolation

Policy: `canonical+strict-murcko+core-topology+bounded-tautomer-v1`.
The local source search found a neutral tautomer of an already reserved compound
whose canonical SMILES and ordinary Murcko string both differed. Neither string
alone can certify an independent molecular population.

Source SMILES, stereo, charge, isotopes, labels and regression/encoder inputs remain
unchanged. Grouping adds aliases to the existing canonical identity and strict
Murcko links; it never removes a legacy connection. Connected components retain
transitive identity/family/series relationships. All acyclic compounds still share
one scaffold group. No random-split fallback is available.

An additional conservative core key retains atom elements and connectivity, removes
terminal non-ring Murcko decorations and ignores bond order, hydrogen placement,
stereo, isotope and formal charge **in the grouping key only**. Thus keto/enol
scaffold bridges are protected even when the bridging molecule is absent. Saturated
and aromatic versions of the same element-labelled core also share a group. These
opaque keys are not physical SMILES, chemical features or a claim that all grouped
molecules have the same HSP. This deliberately coarsens partitions and reservations;
report the resulting loss of rows/families instead of weakening the split.

RDKit tautomer enumeration supplies another identity/scaffold alias, bounded at 256
tautomers and 1,024 transformations. Its status must be `Completed`; partial results
cannot choose a parent. A generated parent must pass the unchanged canonical
parse/serialize identity check. The alias does not neutralize or replace the input.
SolQuest preparation records uncertifiable rows as `incomplete_tautomer_grouping`,
and rejects known reserved topology before enumeration when possible.

The same policy operates in reservation filtering, auxiliary fitting, connected
train/calibration/test grouping and evaluation overlap checks. Programmatic HSP
training checks supplied groups before feature calculation/fitting; coarser groups
are permitted, split aliases are refused. `train_model(..., split_strategy=...)`
must match the caller's scaffold or polymer-series partition. Policy provenance
survives model/split/calibration serialization. Old calibration records must be
recalibrated before evaluation, and inference does not expose obsolete intervals.

## Historical local models

The earlier EGP/GDB17 and QM9 artifacts preserve their original strict-Murcko plans,
source hashes, predictions and metrics. They are not silently rewritten as fits
under the stronger policy. In the EGP audit, 1,500/168/1,052 historical train/
calibration/test rows intersect the stronger reservations; four calibration rows
cannot complete grouping. 142 aliases cross partitions, involving 4,780 rows.
These counts overlap and must not be added as independent observations.

The prior narrower tautomer-only audit already found 28 affected EGP training rows.
Its GDB17 audit was explicitly superseded before completion when core-topology
bridges were added. The completed GDB17 audit finds 6,187/2,315/3,295 reserved
historical train/calibration/test rows and 67/23/11 rows with incomplete grouping.
11,445 aliases cross partitions, involving 104,394 rows. These counts also overlap.
The [QM9 reconstruction and audit](qm9-current-policy.md) also finds 10,006 affected
training rows and old partition links, requiring a new selection and refit.
No historical head is promoted as a leakage-isolated transfer model.
Filtering only test rows cannot repair conflicting training: prepare a new
structure-only population, freeze a new plan and refit. Exact old metrics remain
historical diagnostics, not evidence of 1,000-compound HSP parity.

## Current-policy refits

The new preparation uses only structures for eligibility and connected grouping.
Original source targets, chemical features and frozen encoder inputs stay unchanged.
From the original 8,268 EGP structures, 5,544 remain in 682 groups; 2,720 reserved
rows and four incomplete enumerations are excluded. From 189,764 GDB17 structures,
159,378 remain in 77,358 groups; 16,171 HSP-reserved rows, 14,120 additional EGP-core
rows and 95 incomplete enumerations are excluded. GDB17 reserves all 730 EGP core
keys, including the EGP records whose tautomer enumeration cannot complete.
These exclusion categories are sequential, disjoint and reconcile each population.

Separate fitting reviews bind new immutable selection and experiment hashes. Full
cached aliases are checked against HSP reservations; GDB17 also receives a full
selected-structure check against EGP cores before fitting. New partitions retain
all alias links, rather than reusing the historical train/calibration/test indices.
See the [new A/B/C comparison](solvation-auxiliary.md) for split counts and results.
Saved heads, predictions, per-solvent metrics and calibration coverage remain local.
The experiments are exploratory because historical source results were already
inspected; they are not newly blinded benchmarks or certified HSP transfer models.

Tests cover heteroaromatic and keto/enol aliases, missing bridge rows, legacy/series
transitivity, direct embargo construction, preserved input identities/charges/heavy
isotopes, incomplete enumeration, caller-supplied split rejection, preparation row
trace, train/calibration/evaluation isolation and obsolete interval refusal. All
fixtures use original structures and arbitrary values; no external table is bundled.
