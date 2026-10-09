"""Deny by default; file manifests are declarations that still need human rights review."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schema import StrictModel

LabelKind = Literal["synthetic", "teacher_reproduction", "experimental"]
UNITS = "MPa^0.5"
BLOCKED_NAMES = {"stefanis_data2.xlsx", "fitting_data.xlsx", "hsp_smiles.csv"}
PERMISSIVE_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0"}


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DatasetManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_kind: LabelKind
    units: Literal["MPa^0.5"] = UNITS
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rights_status: Literal["approved", "pending", "denied"]
    permission_evidence: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    local_evaluation_allowed: bool = False
    training_allowed: bool = False
    derived_labels_allowed: bool = False
    derived_weights_allowed: bool = False
    redistribution_allowed: bool = False
    confidential: bool = False
    temperature_k: float = Field(default=298.15, gt=0)
    teacher_id: str | None = None
    independent_measurements: bool = False

    @model_validator(mode="after")
    def label_contract(self):
        if self.label_kind == "teacher_reproduction" and not self.teacher_id:
            raise ValueError("Teacher reproduction requires teacher_id")
        if self.label_kind == "experimental" and not self.independent_measurements:
            raise ValueError("Experimental evaluation requires independent measurements")
        if self.label_kind != "teacher_reproduction" and self.teacher_id:
            raise ValueError("teacher_id belongs only to teacher reproduction datasets")
        if self.label_kind != "experimental" and self.independent_measurements:
            raise ValueError("Synthetic or teacher labels are not independent measurements")
        return self

    def authorize(self, purpose: Literal["train", "evaluate"]) -> None:
        if self.rights_status != "approved":
            raise ValueError("Dataset rights are not approved")
        if purpose == "train" and not (self.training_allowed and self.derived_weights_allowed):
            raise ValueError("Training and derived-weight permissions are required")
        if purpose == "evaluate" and not self.local_evaluation_allowed:
            raise ValueError("Local evaluation permission is required")
        if self.label_kind == "teacher_reproduction" and not self.derived_labels_allowed:
            raise ValueError("Teacher pseudo-label permission is required")


class EmbeddingManifest(StrictModel):
    schema_version: Literal[1] = 1
    encoder_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_license: str
    code_license: str
    dependency_review: Literal["approved", "pending", "denied"]
    rights_status: Literal["approved", "pending", "denied"]
    permission_evidence: str = Field(min_length=1)
    frozen: Literal[True]
    training_allowed: bool = False
    derived_weights_allowed: bool = False
    commercial_use_allowed: bool = False

    def authorize(self):
        if self.rights_status != "approved" or self.dependency_review != "approved":
            raise ValueError("Embedding weights/code/dependencies are not approved")
        if self.weights_license not in PERMISSIVE_LICENSES:
            raise ValueError("Weight license is not an approved permissive license")
        if self.code_license not in PERMISSIVE_LICENSES:
            raise ValueError("Encoder code license is not an approved permissive license")
        if not all(
            (self.training_allowed, self.derived_weights_allowed, self.commercial_use_allowed)
        ):
            raise ValueError(
                "Embedding training/derived-weight/commercial permissions are required"
            )
