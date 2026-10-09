"""Small fixed-encoder interface; pretrained model downloads are deliberately absent."""

import csv
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .chemistry import normalize_smiles
from .provenance import EmbeddingManifest, file_hash


class Encoder(Protocol):
    encoder_id: str

    def transform(self, smiles: list[str] | tuple[str, ...]) -> np.ndarray: ...

    def metadata(self) -> dict: ...


class MorganEncoder:
    encoder_id = "morgan-radius2-256-v1"

    def transform(self, smiles):
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
        return np.stack(
            [
                generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(normalize_smiles(s)))
                for s in smiles
            ]
        ).astype(float)

    def metadata(self):
        return {"encoder_id": self.encoder_id, "pretrained": False, "fixed": True}


class LocalEmbeddingEncoder:
    """Licensed, precomputed frozen vectors. No code execution/network; unknown SMILES fail."""

    def __init__(self, path: str | Path, manifest_path: str | Path):
        path = Path(path)
        self.manifest = EmbeddingManifest.model_validate_json(
            Path(manifest_path).read_text(encoding="utf-8")
        )
        self.manifest.authorize()
        if file_hash(path) != self.manifest.sha256:
            raise ValueError("Embedding checksum mismatch")
        self.encoder_id = self.manifest.encoder_id
        self.vectors = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if (
                len(fields) < 2
                or fields[0] != "smiles"
                or fields[1:] != [f"e{i}" for i in range(len(fields) - 1)]
            ):
                raise ValueError("Embedding CSV needs smiles,e0,e1,... columns")
            for row in reader:
                key = normalize_smiles(row["smiles"])
                vector = np.array([float(row[f]) for f in fields[1:]], dtype=float)
                if key in self.vectors or not np.isfinite(vector).all():
                    raise ValueError("Duplicate canonical SMILES or nonfinite embedding")
                self.vectors[key] = vector
        if not self.vectors:
            raise ValueError("Empty embedding table")

    def transform(self, smiles):
        try:
            return np.stack([self.vectors[normalize_smiles(s)] for s in smiles])
        except KeyError as exc:
            raise ValueError("Missing local frozen embedding; no fallback or download") from exc

    def metadata(self):
        return {
            "encoder_id": self.encoder_id,
            "pretrained": True,
            "manifest": self.manifest.model_dump(),
        }


def get_encoder(name="morgan", path=None, manifest_path=None) -> Encoder:
    if name == "morgan":
        return MorganEncoder()
    if name == "local":
        if not path or not manifest_path:
            raise ValueError("Local embeddings need a CSV and an approved embedding manifest")
        return LocalEmbeddingEncoder(path, manifest_path)
    if name == "molformer":
        raise ValueError(
            "MoLFormer is disabled pending exact weight/revision/dependency review; see issue #3"
        )
    raise ValueError("Unknown encoder")


class PhysicsBackend(Protocol):
    """Future local MD backend; schema is not a claim of a valid HSP decomposition."""

    backend_id: str

    def simulate(self, input_record: dict, settings: dict) -> dict: ...

    def provenance(self) -> dict: ...


def export_encoder_metadata(encoder: Encoder) -> str:
    return json.dumps(encoder.metadata(), sort_keys=True)
