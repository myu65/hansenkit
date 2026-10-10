# Local experiment protocol, 2026-10-10

The user assumes audits passed for local research. This is recorded as an operator assumption,
not a new license or authority to publish third-party datasets or trained weights. Retain exact
source receipts, hashes, model revisions, local manifests and numerical reports outside Git.

## Targets and success criteria

The requested result is at least 1,000 chemically distinct compounds with three HSP predictions
of accuracy comparable to an established method, robust polymer/surfactant computation, and
at least 10,000 training records where available. This goal is **not achieved yet**.
Count supervised HSP, teacher HSP, quantum auxiliary and encoder pretraining records separately.
Training on 100,000 quantum properties does not supply 100,000 HSP measurements.

Compare on identical eligible compounds, report the eligible denominator, every refusal/failure,
three-component MAE/RMSE/R², scaffold/series partitions and OOD errors. For an initial
noninferiority comparison, predeclare an MAE and RMSE ratio ceiling of 1.10 per component against
the named comparator, plus a paired scaffold bootstrap 95% upper difference bound of
0.30 MPa^0.5 for MAE. These are research acceptance criteria, not a safety or formulation claim.
Do not choose success thresholds or select the best model after seeing final test errors.
An upstream GC implementation is a reproducibility comparator until its mathematical and
atom-coverage behavior has been checked against the original method.

## Isolation

1. HSPiT workbook numbers, including coefficient tables, are evaluation-only. They cannot
   supply training labels, residual targets, fitted features, tuning or calibration.
2. Reserve workbook canonical identities and Murcko families before any fitting. The strict
   current implementation includes all acyclic compounds in one group. Report the severe
   loss of training coverage explicitly; do not quietly switch to a random split.
3. Before quantum auxiliary training, embargo **all** HSP_SMILES identities and scaffolds as
   well as the Excel identities. This also reserves the future HSP evaluation population.
4. A fixed, previously pretrained MoLFormer may embed holdouts without fitting or using their
   labels. Its original pretraining corpus may include their structures; disclose this inherited
   limitation. Train-only scaling and regression remain mandatory.
5. Published compilations without verified measurement/estimated lineage are
   `published_reference`, not independent experimental truth. A 10,000-row solubility table
   with repeated temperature/composition measurements is not 10,000 unique HSP compounds.

## First auxiliary run

- DeepChem QM9 CSV: 133,885 rows, six computed quantum targets (`mu`, `alpha`, `gap`, `cv`,
  `u0`, `r2`). Explicit target units are recorded; none is an HSP label.
- After strict HSP/Excel identity and scaffold embargo and canonical deduplication:
  90,643 eligible quantum records, 15,926 scaffold groups; 43,234 source rows excluded by embargo.
- Group split precedes fitted preprocessing. Frozen MoLFormer 768-dimensional vectors feed
  a ridge head with alpha 100 selected before test evaluation. Report actual training and
  held-out counts after fitting. Do not count the entire eligible pool as training.
- Checkpoint files remain identical to the reviewed immutable release. CUDA 12.6 runtime is
  isolated in a local virtual environment. Select the RTX 2080 Ti by UUID, because CUDA device
  numbering differs from `nvidia-smi` on this host; confirm the device name before allocation.
- An initial control included uncharacterized structures; the definitive run removes those
  identities using the original consistency-failure list before refitting.

The original 3,054 inconsistent-geometry list is available through the URL used by
[PyTorch Geometric's QM9 loader](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/qm9.py).
The filter removed another 2,044 eligible rows, leaving 88,599 records. The definitive grouped
fit used **69,142 training rows and 19,457 test rows**, with 12,478 / 3,120 scaffold groups and
zero group overlap. The saved JSON head was reloaded, and all six reported test MAE/RMSE
values were recomputed from the saved fixed vectors and matched. This filter addresses
structure/label consistency, not liquid-phase physics or HSP validation.

## Running a prepared local auxiliary dataset

Provide an NPZ with Unicode `smiles`, a two-dimensional numeric `targets` array and
Unicode `target_names`, plus a checksum-bound [manifest template](../examples/auxiliary-manifest.template.json).
The manifest must document source-specific rights, units and all filtering. Put a JSON object
with canonical reserved HSP `smiles` and optional reserved `series` in a separate local file.
Embargo verification and scaffold splitting happen before encoding/scaling/regression.

```sh
uv run hansenkit auxiliary-fit --data local/quantum-clean.npz --manifest local/quantum-manifest.json --reserved-identities local/hsp-holdouts.json --encoder molformer --checkpoint local/molformer --checkpoint-review local/molformer-runtime-review.json --encoder-device cuda --model models/quantum-head.json --report runs/quantum-report.json
```

Use an isolated CUDA environment matching the explicit checkpoint review. The default
locked MoLFormer extra installs CPU PyTorch. No automatic data/checkpoint download occurs.
Auxiliary heads are separate JSON artifacts and are not accepted as HSP prediction models.

## Initial 1,030-compound reference experiment

The existing HSPiT implementation produced finite output on 1,030 distinct connected net-neutral
organic structures from the published compilation, with 114 calculation failures recorded
separately. Fixed MoLFormer embeddings were available for all 1,030. These structures are the
initial comparison population; this is not a proven independent measured dataset.

Before fitting, an exploratory five-fold scaffold plan recorded identities, folds, alpha candidates,
features and nonnegative clipping. All Excel identities and scaffolds were permanently embargoed
from every supervised fold and every inner selection/cross-fit. Only 176 reference rows were
eligible for supervised HSP fitting; individual outer folds used 113–176. No workbook numerical
value or GC prediction supplied any training feature, target or residual. Full reports, predictions,
saved heads and frozen plans remain local. Public model release is not authorized.

| Registered candidate / comparator | δD MAE | δP MAE | δH MAE |
| --- | ---: | ---: | ---: |
| A: original group and descriptor densities + ridge | 0.993 | 2.992 | 1.868 |
| B: frozen MoLFormer + ridge | 1.301 | 3.027 | 3.071 |
| C: A + group cross-fitted embedding residual | 0.971 | 3.205 | 2.491 |
| A + separately learned quantum features | 0.945 | 3.012 | 1.864 |
| Unmodified HSPiT Stefanis GC implementation | 0.984 | 2.868 | 2.461 |

Units: MPa^0.5. Parameters were chosen by training-only inner group validation, not outer test
metrics. This is an exploratory comparison of all registered candidates, not post-test selection
of a proven final model. The quantum head adds little beyond intensive chemistry features in
this run; MoLFormer-only and hybrid results do not establish superiority.
The GC implementation still needs original-equation and counting validation before it can
represent the established published method. [Related-source and uncertainty review](related-reference-review.md)
records 792 OOD rows and the failed δP group-bootstrap noninferiority criterion.
Extrapolation, predictive interval calibration and polymer/surfactant validation remain required.
These errors must not be described
as independently verified experimental accuracy or as fulfillment of the complete goal.

## Remaining work

Verify existing GC atom counting and original equations; establish the 1,000-compound HSP
comparison; compare A/B/C with train-only selection and independent Excel evaluation; test
auxiliary transfer against a no-auxiliary control. Add qualified polymer/end-group/EO–PO
assembly and a separately validated intensive-property backend. Numerical stability alone
does not establish physical validity for polymers, ions or mixtures.
