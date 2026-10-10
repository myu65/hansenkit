# Published comparator and reference identity audit

Local review, 2026-10-10. Related research assets remain outside this repository.
The user's audit assumption permits this local experiment; it is not permission to
redistribute an external table, source implementation, or fitted research weights.

## Source and reference lineage

[Mathieu (2018)](https://doi.org/10.1021/acsomega.8b02601) provides a Python implementation
and a spreadsheet with reference/calculated values and component-specific split flags.
The article describes 174 compounds with experimentally confirmed references for fitting
and 769 additional compounds with previously estimated references for validation.
The spreadsheet contains 943 compound records, not 943 independently measured test labels.

For D/P/H respectively, flags 0 (source calibration) occur 151/144/153 times;
flags 1 (source validation) occur 707/545/707 times; flags 2 (outside the source applicability
domain) occur 85/254/83 times. A flag 2 alone does not identify measurement versus estimation.
Source calibration records are not a held-out test of that published model. Preserve these
distinctions rather than converting the whole spreadsheet into `experimental` labels.

The [program distribution page](https://figshare.com/articles/dataset/7449257) displays
CC BY-NC 4.0. Its ZIP and spreadsheet were obtained through the publisher's public download
controls. Local checksums:

| Asset | SHA-256 |
| --- | --- |
| `ao8b02601_si_003.zip` | `6fc9f6ec22ce29d083ebdb3070fea2dfb067570593d63f0a2415ed4d122a1338` |
| Extracted `calcHSPs.py` | `7389ff3a95d35fb8ddf2eec147a4e715ed66db6ef911dac4cd907472a2fec0c8` |
| `ao8b02601_si_002.xlsx` | `7b6c78f00863288fcd9afb3409c81b6837cf24d0bedc0f51bac53c508becda53` |

Neither coefficients nor reference/calculated spreadsheet values supplied training,
features, pseudo-labels, hyperparameter selection, or calibration in this comparison.
Keep the supplementary spreadsheet evaluation-only. Future fitting must reserve its
verified identities/families before fitting; unresolved identities must be reconciled first.

## Common-row calculation

The unchanged supplementary program computes all three components for 849 of the current
1,030 reference structures. The other 181 records remain failures because source atom/group
parameters are missing or unusable. They are not silently dropped from a coverage claim.
The following development comparison uses the same 849 finite rows for every method.

| Method | D MAE | P MAE | H MAE |
| --- | ---: | ---: | ---: |
| Original supplementary code | 0.6421 | 2.0741 | 1.3880 |
| Initial intensive Ridge, v1 | 0.9668 | 2.8720 | 1.8127 |
| Initial intensive Ridge + quantum auxiliary features | 0.9160 | 2.9054 | 1.8115 |
| Geometry + fixed LightGBM development candidate | 1.0773 | 2.2420 | 2.0023 |

Units are MPa^0.5. Source overlap, uncertain reference lineage, input identity errors and
reuse of a development population prevent an independent accuracy claim. This does not
establish a winning final model or completion of the 1,000-compound accuracy objective.
The earlier comparison against HSPiT is retained; it cannot substitute for validation of
the original established method. All evaluated candidates and negative findings remain.

## CAS/structure errors

CAS joining matches 787 of the current structures to the supplementary table; 604 reference
triples agree within 0.051. Agreement alone does not verify molecular identity. Among 655
rows with both table calculations and finite code output, 647 agree within that rounding
threshold. Large discrepancies led to checking the structures rather than changing coefficients.

Two confirmed examples in the original HSP reference CSV:

- CAS 460-19-5 is [cyanogen, PubChem CID 9999](https://pubchem.ncbi.nlm.nih.gov/compound/9999),
  but the CSV assigns an ethylenediamine structure.
- CAS 1071-98-3 is [dicyanoacetylene, PubChem CID 14068](https://pubchem.ncbi.nlm.nih.gov/compound/14068),
  but the CSV assigns a saturated diamine structure.

Using the independently confirmed nitrile structures reproduces the source table's rounded
calculations for these examples. A secondary-versus-primary amine mismatch is also under
review. Do not infer that only these records are wrong: the full CAS/structure audit is
tracked in [#15](https://github.com/myu65/hansenkit/issues/15). Raw data and previous results
are preserved. Corrections must follow source identity evidence, not prediction error.

The [expanded identity audit](identity-audit.md) now includes complete batch correspondence,
aggregate discrepancy classifications and the larger Excel reservation. The historical
comparison above is retained with its original inputs; no independent accuracy claim is added.

[PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) supports the local identity
audit. Its request plan contains identifiers only, retains all returned CID candidates,
uses at most two concurrent requests, and does not silently select the first hit or
bypass TLS checks. Ambiguous, unmatched, stereo and tautomer cases need explicit classification.
Freeze the reconciled data version and prospective evaluation plan before the next fit.
