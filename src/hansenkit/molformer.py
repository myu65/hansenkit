"""Opt-in frozen encoder. Only the audited publisher revision can execute, offline."""

import hashlib
import importlib.util
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
from pydantic import Field
from rdkit import Chem

from .chemistry import normalize_smiles
from .provenance import file_hash
from .schema import StrictModel

MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"
# These are reviewed source/tokenizer/weight bytes, not a caller-controlled code allowlist.
APPROVED_FILES = {
    "README.md": "fff331c6a973fa6995662ac31354c702ac0f4805d494e7506b979a3a6cee6f44",
    "config.json": "3ef9eaac8c7ca6282fd6256ed038d151bd4ff42a4ff855367e0d7197bbc1c284",
    "configuration_molformer.py": (
        "b88ea8d4b7b5e54f4f186cc7a230eff308928020a030f153a038fd12c05e3bed"
    ),
    "modeling_molformer.py": "6f1ef72022de2c69e95661899422a7bb39a40a2cc5a6cb6216f14e9b7d84559c",
    "model.safetensors": "0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6",
    "tokenizer.json": "3df1f2219653c44fac9fa03b7f788b372eb2544ecc176737bb9aca8411b471a5",
    "tokenizer_config.json": "56f93dc43e4383fcfc6342e09d782ef38d48ef227215a4a92ff8762fd5b8ff3e",
}
RUNTIME_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")


class CheckpointReview(StrictModel):
    schema_version: Literal[1] = 1
    model_id: Literal[MODEL_ID] = MODEL_ID
    revision: Literal[REVISION] = REVISION
    files: dict[str, str]
    weights_license: Literal["Apache-2.0"] = "Apache-2.0"
    code_license: Literal["Apache-2.0"] = "Apache-2.0"
    rights_status: Literal["approved", "pending", "denied"]
    dependency_review: Literal["approved", "pending", "denied"]
    reviewed_local_code_execution_allowed: bool = False
    training_allowed: bool = False
    derived_weights_allowed: bool = False
    commercial_use_allowed: bool = False
    permission_evidence: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    runtime_versions: dict[str, str]

    def authorize(self):
        if self.files != APPROVED_FILES:
            raise ValueError("Checkpoint revision/files differ from the audited publisher release")
        if self.rights_status != "approved" or self.dependency_review != "approved":
            raise ValueError("Checkpoint rights or dependencies are not approved")
        if not all(
            (
                self.reviewed_local_code_execution_allowed,
                self.training_allowed,
                self.derived_weights_allowed,
                self.commercial_use_allowed,
            )
        ):
            raise ValueError(
                "Checkpoint execution/training/derived-weight permissions are required"
            )
        if set(self.runtime_versions) != set(RUNTIME_PACKAGES):
            raise ValueError("Exact versions for all encoder runtime packages are required")


def verify_checkpoint(directory: Path, review: CheckpointReview, check_runtime=True):
    review.authorize()
    directory = directory.resolve()
    for name, expected in APPROVED_FILES.items():
        path = directory / name
        if path.is_symlink() or path.resolve().parent != directory or not path.is_file():
            raise ValueError("Checkpoint file is missing or escapes the reviewed directory")
        if file_hash(path) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    if check_runtime:
        try:
            if any(version(pkg) != review.runtime_versions[pkg] for pkg in RUNTIME_PACKAGES):
                raise ValueError("Encoder runtime versions differ from the reviewed environment")
        except PackageNotFoundError as exc:
            raise ImportError(
                "Install the optional molformer extra in the locked environment"
            ) from exc


def fetch_checkpoint(directory, review_path):
    """Explicit network operation, called only by the checkpoint-fetch command."""
    review = CheckpointReview.model_validate_json(Path(review_path).read_text(encoding="utf-8"))
    review.authorize()
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Checkpoint output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in APPROVED_FILES.items():
        path = directory / name
        partial = directory / f"{name}.part"
        if path.resolve().parent != directory or partial.resolve().parent != directory:
            raise ValueError("Checkpoint destination escapes the requested directory")
        request = urllib.request.Request(
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
            headers={"User-Agent": "hansenkit-explicit-checkpoint-fetch/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as source, partial.open("xb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        if file_hash(partial) != expected:
            raise ValueError(f"Downloaded checkpoint checksum mismatch: {name}")
        partial.replace(path)
    verify_checkpoint(directory, review, check_runtime=False)
    return directory


def nonisomeric_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(normalize_smiles(smiles)), isomericSmiles=False)


def load_encoder_tensors(model, path):
    """Load and compare every encoder parameter/buffer to the reviewed Safetensors bytes."""
    import torch
    from safetensors.torch import load_file

    raw = load_file(str(path), device="cpu")
    if any(not key.startswith(("molformer.", "lm_head.")) for key in raw):
        raise ValueError("Unexpected checkpoint tensor namespace")
    state = {
        key.removeprefix("molformer."): value
        for key, value in raw.items()
        if key.startswith("molformer.")
    }
    if set(state) != set(model.state_dict()):
        raise ValueError("Checkpoint encoder parameters/buffers do not match the architecture")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint tensor shapes or dtypes are incompatible") from exc
    if any(
        tensor.dtype != state[key].dtype or not torch.equal(tensor, state[key])
        for key, tensor in model.state_dict().items()
    ):
        raise ValueError("Encoder parameters differ from the pretrained checkpoint after loading")


def _load_reviewed_source(full_name, path, expected_sha256):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Reviewed source changed before execution")
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    # Compile the verified bytes, rather than trusting an existing __pycache__ entry.
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class FrozenMolformerEncoder:
    encoder_id = MODEL_ID

    def __init__(self, directory, review_path, batch_size=16, device="cpu"):
        if batch_size < 1 or device not in {"cpu", "cuda"}:
            raise ValueError("Invalid encoder batch size or device")
        self.directory = Path(directory).resolve()
        self.review = CheckpointReview.model_validate_json(
            Path(review_path).read_text(encoding="utf-8")
        )
        verify_checkpoint(self.directory, self.review)
        self.batch_size, self.device = batch_size, device
        self._model = self._tokenizer = None
        self._cache = {}

    def metadata(self):
        return {
            "encoder_id": self.encoder_id,
            "pretrained": True,
            "fixed": True,
            "review": self.review.model_dump(),
            "pooling": "masked-mean",
            "preprocessing": "rdkit-canonical-nonisomeric-v1",
            "deterministic_eval": True,
            "implementation": "hash-verified-direct-tensors-v2",
        }

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import PreTrainedTokenizerFast

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable in this encoder environment")
        # Import two hash-verified local files. No AutoModel remote-code loader or HF module cache.
        package_name = f"_hansenkit_reviewed_molformer_{REVISION}"
        package = ModuleType(package_name)
        package.__path__ = [str(self.directory)]
        sys.modules[package_name] = package
        modules = {}
        for name in ("configuration_molformer", "modeling_molformer"):
            full_name = f"{package_name}.{name}"
            modules[name] = _load_reviewed_source(
                full_name, self.directory / f"{name}.py", APPROVED_FILES[f"{name}.py"]
            )
        config = modules["configuration_molformer"].MolformerConfig.from_pretrained(
            str(self.directory),
            local_files_only=True,
            deterministic_eval=True,
        )
        config._attn_implementation = "eager"
        model = modules["modeling_molformer"].MolformerModel(config).float()
        load_encoder_tensors(model, self.directory / "model.safetensors")
        model.eval()
        model.requires_grad_(False)
        self._model = model.to(self.device)
        self._tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(self.directory),
            local_files_only=True,
        )

    def transform(self, smiles):
        if not smiles:
            raise ValueError("Encoder input cannot be empty")
        canonical = [nonisomeric_smiles(s) for s in smiles]
        missing = list(dict.fromkeys(s for s in canonical if s not in self._cache))
        if missing:
            self._load()
            import torch

            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                inputs = self._tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
                if inputs["input_ids"].shape[1] > self._model.config.max_position_embeddings:
                    raise ValueError(
                        "SMILES exceeds the reviewed encoder token limit; no truncation"
                    )
                if (inputs["input_ids"] == self._tokenizer.unk_token_id).any():
                    raise ValueError("SMILES contains a token outside the encoder vocabulary")
                inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in inputs.items()
                    if name in {"input_ids", "attention_mask"}
                }
                with torch.inference_mode():
                    vectors = self._model(**inputs).pooler_output.cpu().numpy()
                if vectors.shape != (len(batch), 768) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid frozen encoder output")
                self._cache.update(zip(batch, vectors, strict=True))
        return np.stack([self._cache[s] for s in canonical])
