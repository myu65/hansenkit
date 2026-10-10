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
bridges were added. The complete new-policy GDB17 audit is pending at this snapshot.
QM9 also needs its original training population reconstructed and audited under
this policy. No historical head is promoted as a leakage-isolated transfer model.
Filtering only test rows cannot repair conflicting training: prepare a new
structure-only population, freeze a new plan and refit. Exact old metrics remain
historical diagnostics, not evidence of 1,000-compound HSP parity.

Tests cover heteroaromatic and keto/enol aliases, missing bridge rows, legacy/series
transitivity, direct embargo construction, preserved input identities/charges/heavy
isotopes, incomplete enumeration, caller-supplied split rejection, preparation row
trace, train/calibration/evaluation isolation and obsolete interval refusal. All
fixtures use original structures and arbitrary values; no external table is bundled.
