"""JSON model artifacts; no pickle and no implicitly downloaded encoder weights."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .chemistry import FEATURE_VERSION, chemical_features
from .data import Dataset, write_json
from .encoders import Encoder, MorganEncoder
from .provenance import UNITS, DatasetManifest
from .schema import MoleculeInput
from .scope import assess_scope


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, x, y, alpha=1.0):
        scaler = StandardScaler().fit(x)
        regressor = Ridge(alpha=alpha).fit(scaler.transform(x), y)
        return cls(scaler.mean_, scaler.scale_, regressor.coef_, regressor.intercept_)

    def predict(self, x):
        return ((x - self.mean) / self.scale) @ self.coef.T + self.intercept

    def state(self):
        return {
            "kind": "ridge",
            **{k: getattr(self, k).tolist() for k in ("mean", "scale", "coef", "intercept")},
        }

    @classmethod
    def restore(cls, state):
        return cls(
            **{k: np.array(state[k], dtype=float) for k in ("mean", "scale", "coef", "intercept")}
        )


class LightGBMHead:
    def __init__(self, boosters):
        self.boosters = boosters

    @classmethod
    def fit(cls, x, y, seed=42):
        import lightgbm as lgb

        boosters = []
        for i in range(3):
            estimator = lgb.LGBMRegressor(
                n_estimators=80,
                num_leaves=7,
                min_child_samples=8,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            ).fit(x, y[:, i])
            boosters.append(estimator.booster_)
        return cls(boosters)

    def predict(self, x):
        return np.column_stack([b.predict(x) for b in self.boosters])

    def state(self):
        return {"kind": "lightgbm", "models": [b.model_to_string() for b in self.boosters]}

    @classmethod
    def restore(cls, state):
        import lightgbm as lgb

        return cls([lgb.Booster(model_str=s) for s in state["models"]])


def restore_head(state):
    if state["kind"] == "ridge":
        return LinearHead.restore(state)
    if state["kind"] == "lightgbm":
        return LightGBMHead.restore(state)
    raise ValueError("Unknown model head")


def feature_matrix(smiles):
    return np.stack([chemical_features(s) for s in smiles])


@dataclass
class HSPModel:
    mode: str
    head: LinearHead | LightGBMHead
    residual: LinearHead | None
    encoder_metadata: dict
    provenance: dict
    train_smiles: tuple[str, ...]
    feature_min: np.ndarray
    feature_max: np.ndarray
    conformal_radius: np.ndarray | None = None
    calibration_info: dict | None = None

    def predict(self, smiles, encoder: Encoder | None = None):
        # Evaluation needs raw OOD scores; chemically unsupported inputs still always fail.
        if any(not assess_scope(MoleculeInput(smiles=s)).supported for s in smiles):
            raise ValueError("Unsupported structure; no numerical prediction is available")
        if self.mode == "A":
            return self.head.predict(feature_matrix(smiles))
        encoder = encoder or MorganEncoder()
        if encoder.metadata() != self.encoder_metadata:
            raise ValueError("Encoder identity, revision or provenance differs from training")
        e = encoder.transform(smiles)
        if self.mode == "B":
            return self.head.predict(e)
        return self.head.predict(feature_matrix(smiles)) + self.residual.predict(e)

    def save(self, path: str | Path):
        path = Path(path)
        if path.exists():
            raise ValueError("Model output exists; choose a new file")
        write_json(
            path,
            {
                "schema_version": 1,
                "feature_version": FEATURE_VERSION,
                "mode": self.mode,
                "head": self.head.state(),
                "residual": self.residual.state() if self.residual else None,
                "encoder": self.encoder_metadata,
                "provenance": self.provenance,
                "train_smiles": self.train_smiles,
                "feature_min": self.feature_min.tolist(),
                "feature_max": self.feature_max.tolist(),
                "conformal_radius": self.conformal_radius.tolist()
                if self.conformal_radius is not None
                else None,
                "calibration": self.calibration_info,
            },
        )

    @classmethod
    def load(cls, path: str | Path):
        import json

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state["schema_version"] != 1 or state["feature_version"] != FEATURE_VERSION:
            raise ValueError("Model schema or feature version mismatch")
        if state["mode"] not in {"A", "B", "C"}:
            raise ValueError("Unknown model mode")
        manifest = DatasetManifest.model_validate(state["provenance"]["dataset_manifest"])
        manifest.authorize("train")
        if (
            state["provenance"]["label_kind"] != manifest.label_kind
            or state["provenance"]["units"] != UNITS
        ):
            raise ValueError("Model provenance/label kind mismatch")
        trained_versions = state["provenance"]["runtime_versions"]
        packages = ("rdkit", "numpy", "scikit-learn")
        if state["head"]["kind"] == "lightgbm":
            packages += ("lightgbm",)
        if any(version(pkg) != trained_versions[pkg] for pkg in packages):
            raise ValueError(
                "Model dependency versions differ; recreate the locked training environment"
            )
        return cls(
            mode=state["mode"],
            head=restore_head(state["head"]),
            residual=restore_head(state["residual"]) if state["residual"] else None,
            encoder_metadata=state["encoder"],
            provenance=state["provenance"],
            train_smiles=tuple(state["train_smiles"]),
            feature_min=np.array(state["feature_min"]),
            feature_max=np.array(state["feature_max"]),
            conformal_radius=np.array(state["conformal_radius"])
            if state["conformal_radius"] is not None
            else None,
            calibration_info=state["calibration"],
        )


def train_model(
    dataset: Dataset, indices, groups, mode="A", encoder=None, head="ridge", seed=42
) -> HSPModel:
    dataset.manifest.authorize("train")
    if mode not in {"A", "B", "C"}:
        raise ValueError("Mode must be A, B or C")
    if head not in {"ridge", "lightgbm"} or (head == "lightgbm" and mode != "A"):
        raise ValueError("LightGBM is optional for A; B/C use ridge")
    encoder = encoder or MorganEncoder()
    smiles = tuple(dataset.smiles[i] for i in indices)
    x, y = feature_matrix(smiles), dataset.targets[indices]
    residual = None
    if mode == "B":
        fitted = LinearHead.fit(encoder.transform(smiles), y, alpha=5)
    else:
        if head == "lightgbm":
            fitted = LightGBMHead.fit(x, y, seed)
        else:
            fitted = LinearHead.fit(x, y)
        if mode == "C":
            train_groups = groups[indices]
            if len(set(train_groups)) < 3:
                raise ValueError(
                    "Hybrid residual cross-fitting needs at least three training groups"
                )
            oof = np.empty_like(y)
            # Every residual label comes from a chemistry model that excluded its whole group.
            folds = GroupKFold(n_splits=min(4, len(set(train_groups))))
            for inner_train, inner_valid in folds.split(x, y, train_groups):
                oof[inner_valid] = LinearHead.fit(x[inner_train], y[inner_train]).predict(
                    x[inner_valid]
                )
            residual = LinearHead.fit(encoder.transform(smiles), y - oof, alpha=10)
    packages = ("hansenkit", "rdkit", "numpy", "scikit-learn")
    if head == "lightgbm":
        packages += ("lightgbm",)
    provenance = {
        "label_kind": dataset.manifest.label_kind,
        "units": UNITS,
        "dataset_manifest": dataset.manifest.model_dump(),
        "seed": seed,
        "scope": "neutral-small-molecule-298.15K-poc",
        "real_accuracy_validated": False,
        "residual_cross_fitted": mode == "C",
        "runtime_versions": {pkg: version(pkg) for pkg in packages},
    }
    return HSPModel(
        mode, fitted, residual, encoder.metadata(), provenance, smiles, x.min(0), x.max(0)
    )
