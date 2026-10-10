"""Local quantum auxiliary regression. These labels are never called HSP measurements."""

from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from sklearn.model_selection import GroupShuffleSplit

from .data import write_json
from .embargo import HoldoutEmbargo
from .models import LinearHead
from .provenance import file_hash
from .schema import StrictModel
from .splitting import GROUPING_POLICY, grouping_keys


class AuxiliaryManifest(StrictModel):
    source: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_kind: Literal["quantum_computed_auxiliary"]
    target_names: tuple[str, ...] = Field(min_length=1)
    target_units: tuple[str, ...] = Field(min_length=1)
    rights_status: Literal["approved", "pending", "denied"]
    audit_basis: Literal["documented_permission", "operator_assumption"]
    permission_evidence: str = Field(min_length=1)
    training_allowed: bool = False
    derived_weights_allowed: bool = False
    redistribution_allowed: bool = False

    @model_validator(mode="after")
    def check_targets(self):
        if len(self.target_names) != len(self.target_units):
            raise ValueError("Every quantum target needs an explicit unit")
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("Auxiliary target names must be unique")
        if any(n.casefold() in {"delta_d", "delta_p", "delta_h"} for n in self.target_names):
            raise ValueError("HSP targets cannot be relabeled as quantum auxiliary targets")
        if self.audit_basis == "operator_assumption" and self.redistribution_allowed:
            raise ValueError("Operator assumptions cannot authorize redistribution")
        return self

    def authorize(self):
        if self.rights_status != "approved" or not (
            self.training_allowed and self.derived_weights_allowed
        ):
            raise ValueError("Quantum auxiliary training/derived-weight permissions are required")


def fit_auxiliary(path, manifest: AuxiliaryManifest, encoder, embargo: HoldoutEmbargo, seed=42):
    """Fit only after confirming the complete input is isolated from HSP holdouts."""
    manifest.authorize()
    if file_hash(path) != manifest.sha256:
        raise ValueError("Auxiliary checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        if not {"smiles", "targets", "target_names"}.issubset(arrays.files):
            raise ValueError("Auxiliary NPZ needs smiles, targets and target_names arrays")
        if arrays["smiles"].ndim != 1 or arrays["target_names"].ndim != 1:
            raise ValueError("Auxiliary identities and target names must be one-dimensional")
        smiles = tuple(arrays["smiles"].tolist())
        y = np.array(arrays["targets"], dtype=float)
        target_names = tuple(arrays["target_names"].tolist())
    if target_names != manifest.target_names or y.shape != (len(smiles), len(target_names)):
        raise ValueError("Auxiliary target names/shapes disagree with the manifest")
    if len(smiles) < 12 or not np.isfinite(y).all():
        raise ValueError("Insufficient or nonfinite auxiliary records")
    embargo.assert_disjoint(smiles)
    groups = grouping_keys(smiles)
    if len(set(groups)) < 8:
        raise ValueError("Need at least eight independent auxiliary scaffold groups")
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(y, groups=groups)
    )
    # Fixed encoder does not fit on any held-out target. Scaling/regression sees train only.
    x = encoder.transform(smiles)
    head = LinearHead.fit(x[train], y[train], alpha=100)
    prediction = head.predict(x[test])
    error = prediction - y[test]
    report = {
        "task": "quantum_property_auxiliary",
        "grouping_policy": GROUPING_POLICY,
        "hsp_training_rows": 0,
        "quantum_training_rows": len(train),
        "quantum_test_rows": len(test),
        "train_groups": len(set(groups[train])),
        "test_groups": len(set(groups[test])),
        "holdout_embargo": embargo.report(),
        "scaffold_overlap": len(set(groups[train]) & set(groups[test])),
        "seed": seed,
        "target_metrics": {
            name: {
                "mae": float(np.abs(error[:, i]).mean()),
                "rmse": float(np.sqrt((error[:, i] ** 2).mean())),
                "units": manifest.target_units[i],
            }
            for i, name in enumerate(target_names)
        },
        "real_hsp_accuracy_validated": False,
    }
    state = {
        "schema_version": 1,
        "task": "quantum_property_auxiliary",
        "grouping_policy": GROUPING_POLICY,
        "head": head.state(),
        "encoder": encoder.metadata(),
        "manifest": manifest.model_dump(),
        "runtime_versions": {p: version(p) for p in ("numpy", "scikit-learn", "rdkit")},
        "report": report,
    }
    return state, report


def save_auxiliary(state, path):
    if Path(path).exists():
        raise ValueError("Auxiliary model output exists")
    write_json(path, state)
