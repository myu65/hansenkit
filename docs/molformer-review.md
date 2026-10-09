# Audited optional frozen MoLFormer

Audit date: 2026-10-10. This review enables only
`ibm-research/MoLFormer-XL-both-10pct` revision `361063d0ad524ef77cf39b08469f6be770dc550f`.
It is the released 10% PubChem/ZINC checkpoint, not the unavailable full XL checkpoint.

## Rights and exact assets

The [publisher's pinned model card](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/blob/361063d0ad524ef77cf39b08469f6be770dc550f/README.md)
declares Apache-2.0, permits frozen feature extraction, and describes the pretraining corpus.
The two checkpoint Python files have Apache-2.0 notices. Source, config and tokenizer bytes
were inspected and hashed. The Safetensors weight SHA-256 matches the publisher's LFS record:
`0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6` (187,248,784 bytes).
Seven approved file hashes are in [the review record](../assets/molformer-review.json).

This approves use of the declared Apache release and derived regression weights; it does not
claim an independent ownership audit of every underlying PubChem/ZINC molecule. No pretraining
corpus or external HSP labels were downloaded. Unsupervised pretraining may have seen evaluation
molecules; supervised HSP-label/scaffold isolation is a separate check.

Reviewed runtime: torch 2.14.1+cpu, transformers 5.12.1, tokenizers 0.22.2, safetensors 0.8.0,
huggingface-hub 1.33.0. The [dependency inventory](dependency-inventory.json) records installed
package declarations and license-file hashes. Libraries have BSD/Apache grants; transitive
packages include MIT/ISC/PSF/CNRI and MPL-2.0 components (certifi/tqdm). These are external
dependencies, not relabeled MIT or bundled with our source. Binary redistribution and notice
obligations require a separate packaging review. Our original code remains MIT.

## Execution and numerical boundaries

Only explicit `checkpoint-fetch` contacts the publisher. Training/prediction require explicit local
checkpoint/review paths, verify hashes/runtime versions, and load Safetensors offline. The adapter
imports two hash-verified local publisher files; it does not call an AutoModel remote-code loader.
Arbitrary revisions/code hashes are rejected. Imported source does tensor/model operations and
imports torch/Transformers; no subprocess, network, dynamic eval, pickle loader or download was found.
Only verified source bytes are compiled; existing bytecode caches are not trusted as substitutes.

Encoder tensors are loaded directly from Safetensors with exact key/shape matching, and every
parameter/buffer is compared to the checkpoint. This avoids a compatibility loading path that
left random linear parameters in this runtime. Regression artifacts carry implementation
`hash-verified-direct-tensors-v2`; unverified earlier prototypes cannot use this encoder identity.

The encoder uses eval mode, `requires_grad=False` and `deterministic_eval=True`. Pooling is masked
mean; SMILES are canonicalized without isomeric information, matching the publisher convention.
Unknown tokens and sequences longer than 202 tokens fail without truncation. Output is 768
dimensions. Small floating differences across hardware/batches may occur; the real checkpoint
was checked across independent loads and padding/batch sizes. Only regression/residual heads
are fitted, using original synthetic labels.

## Reproduce from a checkout

```sh
uv sync --locked --extra lightgbm --extra molformer
uv run hansenkit checkpoint-fetch --out local/molformer --checkpoint-review assets/molformer-review.json
uv run hansenkit compare --data runs/demo-data/synthetic.csv --manifest runs/demo-data/manifest.json --out models/molformer-demo --encoder molformer --checkpoint local/molformer --checkpoint-review assets/molformer-review.json --include-lightgbm
uv run hansenkit predict --model models/molformer-demo/B-ridge.json --input examples/predict.csv --output runs/molformer-predictions.csv --encoder molformer --checkpoint local/molformer --checkpoint-review assets/molformer-review.json
```

Generate demo data as in the README first. Downloading is opt-in and ~187 MB for weights; the
optional CPU framework is separate. The default has no pretrained downloads. Other checkpoint
revisions and GPU runtime versions need their own review.
