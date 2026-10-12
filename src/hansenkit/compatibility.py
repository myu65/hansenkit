"""Array-only binary compatibility research with explicit caller provenance.

Callers review source assets and supply complete, isolated material-family IDs.
This module fits statistical class probabilities from finite arrays and fixed
solvent categories. It supplies no source reader, encoder or material/HSP backend.
The valid flag reports finite arithmetic and categorical support; OOD is a
training-range diagnostic. Probabilities have no calibration qualification.
"""

import copy
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from scipy.optimize import minimize
from scipy.special import expit

DEFAULT_ALPHAS = (0.1, 1.0, 10.0)
FEATURE_VARIATION_TOLERANCE = 1e-12
OOD_TOLERANCE = 1e-12
GRADIENT_INFINITY_GATE = 1e-5
OPTIMIZER_OPTIONS = {
    "maxiter": 2000,
    "maxfun": 10000,
    "maxls": 40,
    "maxcor": 10,
    "ftol": 0.0,
    "gtol": 1e-9,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class CompatibilityProvenance(BaseModel):
    """An asset-specific declaration; maintainers still review its evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    task: Literal["binary_polymer_compatibility"] = "binary_polymer_compatibility"
    label_kind: Literal[
        "synthetic_polymer_compatibility",
        "published_reference_polymer_compatibility_auxiliary",
        "experimental_polymer_compatibility",
    ]
    dataset_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rights_status: Literal["approved", "pending", "denied"]
    allowed_role: Literal["training", "evaluation_only"] = "training"
    permission_evidence: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    audit_basis: Literal["documented_permission", "operator_assumption"]
    use_scope: Literal["scientific_private_noncommercial"] = "scientific_private_noncommercial"
    training_allowed: StrictBool = False
    derived_weights_allowed: StrictBool = False
    local_evaluation_allowed: StrictBool = False
    independent_measurements: StrictBool = False
    measurement_lineage: Literal["authored", "unknown", "independent_measurements"]
    input_representation: Literal["reviewed_source_unit_arrays", "authored_array_fixture"]
    material_conditions_qualified: Literal[False] = False
    public_redistribution: Literal[False] = False
    commercial_allowed: Literal[False] = False
    calibrated_probability_or_coverage: Literal[False] = False
    material_applicability_qualified: Literal[False] = False
    physical_accuracy_qualified: Literal[False] = False
    HSP_predictions: None = None

    @field_validator("dataset_id", "source", "license", "permission_evidence", "approved_by")
    @classmethod
    def nonblank_evidence(cls, value):
        if not value.strip():
            raise ValueError("Nonblank compatibility source and permission evidence required")
        return value

    @model_validator(mode="after")
    def label_contract(self):
        expected = {
            "synthetic_polymer_compatibility": "authored",
            "published_reference_polymer_compatibility_auxiliary": "unknown",
            "experimental_polymer_compatibility": "independent_measurements",
        }[self.label_kind]
        if self.measurement_lineage != expected:
            raise ValueError("Compatibility label kind and measurement lineage conflict")
        if self.independent_measurements != (expected == "independent_measurements"):
            raise ValueError("Independent-measurement declaration conflicts with label kind")
        return self

    def authorize(self, purpose="train"):
        if self.rights_status != "approved" or not self.derived_weights_allowed:
            raise ValueError("Asset rights and derived-weight permission are required")
        if purpose == "train":
            if self.allowed_role != "training" or not self.training_allowed:
                raise ValueError("Evaluation-only or unapproved labels cannot train")
        elif purpose == "predict":
            if not self.local_evaluation_allowed:
                raise ValueError("Local evaluation permission is required")
        else:
            raise ValueError("Unknown compatibility purpose")


@dataclass(frozen=True)
class CompatibilityRow:
    """Opaque metadata: a material/source-unit pair stays in one supplied family."""

    row_id: str
    family_id: str
    pair_id: str
    solvent_id: str


class CompatibilityFitRefusal(ValueError):
    def __init__(self, reason, diagnostics):
        self.reason = reason
        self.diagnostics = copy.deepcopy(diagnostics)
        super().__init__(reason)


def _rows(rows):
    records = tuple(rows)
    _require(
        records and all(isinstance(r, CompatibilityRow) for r in records), "Typed rows required"
    )
    _require(
        all(
            isinstance(s, str) and s.strip()
            for r in records
            for s in (r.row_id, r.family_id, r.pair_id, r.solvent_id)
        ),
        "Nonempty opaque row/family/pair/solvent IDs required",
    )
    _require(len({r.row_id for r in records}) == len(records), "Unique source row IDs required")
    assignments = {}
    for row in records:
        previous = assignments.setdefault(row.pair_id, row.family_id)
        _require(previous == row.family_id, "Repeated material pair crosses supplied families")
    return records


def family_pair_weights(rows):
    """Equal family totals, equal distinct pair/solvent totals, equal raw repeats."""
    records = _rows(rows)
    family_pairs = {}
    for i, row in enumerate(records):
        family_pairs.setdefault(row.family_id, {}).setdefault(
            (row.pair_id, row.solvent_id), []
        ).append(i)
    weights = np.zeros(len(records), dtype=float)
    for pairs in family_pairs.values():
        for indices in pairs.values():
            weights[indices] = 1.0 / len(family_pairs) / len(pairs) / len(indices)
    return weights


def _matrix(value, n=None, p=None):
    matrix = np.asarray(value, dtype=float)
    _require(
        matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] > 0, "Numeric matrix required"
    )
    _require(
        (n is None or matrix.shape[0] == n) and (p is None or matrix.shape[1] == p),
        "Feature alignment",
    )
    _require(np.isfinite(matrix).all(), "Finite training features required; no encoder imputation")
    return matrix


def _labels(value, n):
    y = np.asarray(value, dtype=float)
    _require(
        y.shape == (n,) and np.isfinite(y).all() and np.isin(y, (0.0, 1.0)).all(),
        "Original binary compatibility labels required",
    )
    return y


def _scale_fit(matrix):
    mean, scale = np.mean(matrix, axis=0), np.std(matrix, axis=0)
    active = scale > FEATURE_VARIATION_TOLERANCE
    scale[~active] = 1.0
    parameters = {
        "raw_columns": matrix.shape[1],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "active": active.tolist(),
        "train_min": np.min(matrix, axis=0).tolist(),
        "train_max": np.max(matrix, axis=0).tolist(),
    }
    return parameters, (matrix[:, active] - mean[active]) / scale[active]


def _design(z, solvents, catalogue):
    index = {s: i for i, s in enumerate(catalogue)}
    categorical = np.zeros((len(solvents), len(catalogue)))
    supported = np.array([s in index for s in solvents], dtype=bool)
    for i, solvent in enumerate(solvents):
        if supported[i]:
            categorical[i, index[solvent]] = 1.0
    return np.column_stack((z, categorical)), supported


def logistic_objective_gradient(parameter, design, labels, weights, alpha):
    """Weighted original-label logistic CE and L2 weights; intercept unpenalized."""
    vector, matrix = np.asarray(parameter, float), np.asarray(design, float)
    y, w = np.asarray(labels, float), np.asarray(weights, float)
    _require(matrix.ndim == 2 and vector.shape == (matrix.shape[1] + 1,), "Objective dimensions")
    _require(
        y.shape == w.shape == (len(matrix),)
        and np.isfinite(matrix).all()
        and np.isfinite(vector).all()
        and np.isfinite(y).all()
        and np.isfinite(w).all()
        and np.isin(y, (0.0, 1.0)).all()
        and np.all(w > 0)
        and np.isfinite(alpha)
        and alpha > 0,
        "Invalid logistic objective inputs",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        eta = matrix @ vector[:-1] + vector[-1]
        error = w * (expit(eta) - y)
        loss = w @ (np.logaddexp(0.0, eta) - y * eta) + 0.5 * alpha * (vector[:-1] @ vector[:-1])
        gradient = np.r_[matrix.T @ error + alpha * vector[:-1], np.sum(error)]
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), "Nonfinite objective or gradient")
    return float(loss), gradient


def _runtime():
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}


def _digest(state):
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _training_digest(matrix, labels, rows):
    metadata = [(r.row_id, r.family_id, r.pair_id, r.solvent_id) for r in rows]
    h = hashlib.sha256(matrix.astype("<f8").tobytes() + labels.astype("<f8").tobytes())
    h.update(json.dumps(metadata, separators=(",", ":")).encode())
    return h.hexdigest()


def _fit(matrix, rows, labels, alpha, provenance):
    _require(np.isfinite(alpha) and alpha > 0, "Positive finite alpha required")
    if len(np.unique(labels)) != 2:
        raise CompatibilityFitRefusal(
            "single_class_training_has_no_finite_unpenalized_logit", {"stage": "before_optimizer"}
        )
    catalogue = sorted({r.solvent_id for r in rows})
    support = {s: len({r.family_id for r in rows if r.solvent_id == s}) for s in catalogue}
    _require(min(support.values()) >= 2, "Each fitted solvent requires two training families")
    parameters, z = _scale_fit(matrix)
    design, _ = _design(z, [r.solvent_id for r in rows], catalogue)
    weights = family_pair_weights(rows)
    prevalence = float(weights @ labels)
    initial = np.r_[np.zeros(design.shape[1]), np.log(prevalence / (1.0 - prevalence))]
    try:
        result = minimize(
            logistic_objective_gradient,
            initial,
            args=(design, labels, weights, alpha),
            jac=True,
            method="L-BFGS-B",
            options=copy.deepcopy(OPTIMIZER_OPTIONS),
        )
        value, gradient = logistic_objective_gradient(result.x, design, labels, weights, alpha)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise CompatibilityFitRefusal(
            "optimizer_exception",
            {"exception_type": type(error).__name__, "single_initialization": True},
        ) from error
    grad_inf = float(np.max(np.abs(gradient)))
    consistent = bool(
        np.isfinite(result.fun) and abs(value - float(result.fun)) <= 1e-12 * max(1.0, abs(value))
    )
    diagnostic = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": value,
        "gradient_inf": grad_inf,
        "objective_consistent": consistent,
        "iterations": int(result.nit),
        "single_initialization": True,
        "optimizer_options": copy.deepcopy(OPTIMIZER_OPTIONS),
    }
    if not (
        diagnostic["success"]
        and np.isfinite(result.x).all()
        and consistent
        and grad_inf <= GRADIENT_INFINITY_GATE
    ):
        raise CompatibilityFitRefusal("optimizer_or_gradient_gate_failed", diagnostic)
    state = {
        "schema_version": 1,
        "implementation": "family_categorical_logistic_v1",
        "task": "binary_polymer_compatibility",
        "alpha": float(alpha),
        "parameters": parameters,
        "solvent_catalogue": catalogue,
        "solvent_training_family_support": support,
        "weights": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "diagnostics": diagnostic,
        "provenance": provenance.model_dump(),
        "runtime": _runtime(),
        "training_rows": len(rows),
        "training_families": len({r.family_id for r in rows}),
        "training_arrays_sha256": _training_digest(matrix, labels, rows),
        "selection": None,
        "calibrated_probability_or_coverage": False,
        "material_applicability_qualified": False,
        "physical_accuracy_qualified": False,
        "HSP_predictions": None,
    }
    state["state_sha256"] = _digest(state)
    return state


def _validate(state):
    _require(
        isinstance(state, dict)
        and state.get("schema_version") == 1
        and state.get("implementation") == "family_categorical_logistic_v1"
        and state.get("task") == "binary_polymer_compatibility",
        "Wrong compatibility state",
    )
    provenance = CompatibilityProvenance.model_validate(state["provenance"])
    provenance.authorize("train")
    _require(state.get("state_sha256") == _digest(state), "Primitive model integrity mismatch")
    _require(state["runtime"] == _runtime(), "NumPy/SciPy runtime differs from saved state")
    _require(np.isfinite(state["alpha"]) and state["alpha"] > 0, "Invalid alpha")
    for key in (
        "calibrated_probability_or_coverage",
        "material_applicability_qualified",
        "physical_accuracy_qualified",
    ):
        _require(state.get(key) is False, "Statistical state cannot promote physical qualification")
    _require(state.get("HSP_predictions") is None, "Compatibility states carry no HSP components")
    params = state["parameters"]
    p = params["raw_columns"]
    _require(type(p) is int and p > 0, "Raw feature dimension")
    for key in ("mean", "scale", "active", "train_min", "train_max"):
        _require(len(params[key]) == p, "Preprocessing dimensions")
    _require(all(type(a) is bool for a in params["active"]), "Boolean active-column mask")
    arrays = [np.asarray(params[key], float) for key in ("mean", "scale", "train_min", "train_max")]
    _require(
        all(a.shape == (p,) and np.isfinite(a).all() for a in arrays)
        and np.all(arrays[1] > 0)
        and np.all(arrays[2] <= arrays[3]),
        "Invalid training-only standardizer",
    )
    catalogue = state["solvent_catalogue"]
    _require(
        catalogue
        and catalogue == sorted(set(catalogue))
        and all(isinstance(s, str) and s.strip() for s in catalogue),
        "Invalid solvent catalogue",
    )
    support = state["solvent_training_family_support"]
    _require(
        set(support) == set(catalogue) and all(type(n) is int and n >= 2 for n in support.values()),
        "Invalid solvent support",
    )
    _require(
        len(state["weights"]) == sum(params["active"]) + len(catalogue)
        and np.isfinite(np.r_[state["weights"], state["intercept"]]).all(),
        "Invalid coefficient state",
    )
    diagnostic = state["diagnostics"]
    _require(
        diagnostic["success"] is True
        and diagnostic["status"] == 0
        and diagnostic["objective_consistent"] is True
        and diagnostic["single_initialization"] is True
        and np.isfinite(diagnostic["objective"])
        and np.isfinite(diagnostic["gradient_inf"])
        and 0 <= diagnostic["gradient_inf"] <= GRADIENT_INFINITY_GATE
        and diagnostic["optimizer_options"] == OPTIMIZER_OPTIONS,
        "Refused or inconsistent optimizer state",
    )
    _require(
        type(state["training_rows"]) is int
        and state["training_rows"] >= 2
        and type(state["training_families"]) is int
        and state["training_families"] >= 2,
        "Training population metadata",
    )
    _require(
        isinstance(state["training_arrays_sha256"], str)
        and len(state["training_arrays_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in state["training_arrays_sha256"]),
        "Training-array fingerprint required",
    )
    selection = state["selection"]
    if selection is not None:
        _require(
            isinstance(selection, dict)
            and selection.get("method") == "training_family_cv"
            and selection.get("held_labels_used") is False
            and type(selection.get("folds")) is int
            and selection["folds"] >= 2
            and type(selection.get("seed")) is int
            and selection.get("family_count") == state["training_families"]
            and selection["family_count"] >= selection["folds"]
            and selection.get("selected_alpha") == state["alpha"],
            "Invalid training-only selection metadata",
        )
        candidates = selection["candidates"]
        _require(isinstance(candidates, list) and candidates, "Selection candidates required")
        _require(
            len({c["alpha"] for c in candidates}) == len(candidates), "Distinct selection alphas"
        )
        for candidate in candidates:
            score = candidate["family_logloss"]
            _require(
                np.isfinite(candidate["alpha"])
                and candidate["alpha"] > 0
                and type(candidate["inadmissible"]) is bool
                and isinstance(candidate["failures"], list),
                "Invalid selection candidate",
            )
            if candidate["inadmissible"]:
                _require(
                    score is None and candidate["failures"], "Failed candidate evidence required"
                )
            else:
                _require(
                    score is not None
                    and np.isfinite(score)
                    and score >= 0
                    and not candidate["failures"],
                    "Admissible candidate score required",
                )
        eligible = [c for c in candidates if not c["inadmissible"]]
        _require(eligible, "No selected admissible candidate")
        chosen = min(eligible, key=lambda c: (c["family_logloss"], candidates.index(c)))
        _require(chosen["alpha"] == state["alpha"], "Selection differs from training score")
    return provenance


class CompatibilityClassifier:
    """Shared source-array logit with regularized categorical solvent intercepts."""

    def __init__(self, state):
        self._state = copy.deepcopy(state)
        _validate(self._state)

    @classmethod
    def fit(cls, features, rows, labels, *, provenance, alpha=1.0):
        _require(
            isinstance(provenance, CompatibilityProvenance), "Explicit typed provenance required"
        )
        provenance.authorize("train")
        records = _rows(rows)
        matrix = _matrix(features, len(records))
        y = _labels(labels, len(records))
        return cls(_fit(matrix, records, y, alpha, provenance))

    @classmethod
    def select(cls, features, rows, labels, *, provenance, alphas=DEFAULT_ALPHAS, folds=5, seed=42):
        """Select by pooled family-balanced OOF log loss from training arrays alone."""
        _require(
            isinstance(provenance, CompatibilityProvenance), "Explicit typed provenance required"
        )
        provenance.authorize("train")
        provenance.authorize("predict")
        records = _rows(rows)
        matrix, y = _matrix(features, len(records)), _labels(labels, len(records))
        candidates = tuple(alphas)
        _require(
            candidates
            and len(set(candidates)) == len(candidates)
            and all(np.isfinite(a) and a > 0 for a in candidates),
            "Positive distinct alphas",
        )
        _require(
            type(folds) is int and folds >= 2 and type(seed) is int, "Integer family-fold settings"
        )
        families = sorted(
            {r.family_id for r in records},
            key=lambda f: (hashlib.sha256(f"compatibility-{seed}\0{f}".encode()).hexdigest(), f),
        )
        _require(len(families) >= folds, "Too few independent supplied training families")
        assignments = {family: i % folds for i, family in enumerate(families)}
        reports = []
        for alpha in candidates:
            oof, failures = np.full(len(records), np.nan), []
            for fold in range(folds):
                tr = np.array(
                    [i for i, r in enumerate(records) if assignments[r.family_id] != fold]
                )
                va = np.array(
                    [i for i, r in enumerate(records) if assignments[r.family_id] == fold]
                )
                try:
                    model = cls.fit(
                        matrix[tr],
                        [records[i] for i in tr],
                        y[tr],
                        provenance=provenance,
                        alpha=alpha,
                    )
                    prediction = model.predict(matrix[va], [records[i].solvent_id for i in va])
                    if not prediction["valid"].all():
                        raise CompatibilityFitRefusal(
                            "validation_solvent_or_numeric_refusal",
                            {"refused_rows": int(np.sum(~prediction["valid"]))},
                        )
                    oof[va] = prediction["logit"]
                except (CompatibilityFitRefusal, ValueError) as error:
                    failures.append(
                        {
                            "fold": fold,
                            "reason": str(error),
                            "diagnostics": copy.deepcopy(getattr(error, "diagnostics", {})),
                        }
                    )
            loss = (
                binary_compatibility_scores(y, oof, records)["logloss"]
                if not failures and np.isfinite(oof).all()
                else None
            )
            reports.append(
                {
                    "alpha": float(alpha),
                    "family_logloss": loss,
                    "inadmissible": loss is None,
                    "failures": failures,
                }
            )
        eligible = [r for r in reports if not r["inadmissible"]]
        if not eligible:
            raise CompatibilityFitRefusal(
                "all_training_family_candidates_inadmissible", {"candidates": reports}
            )
        chosen = min(eligible, key=lambda r: (r["family_logloss"], candidates.index(r["alpha"])))
        model = cls.fit(matrix, records, y, provenance=provenance, alpha=chosen["alpha"])
        model._state["selection"] = {
            "method": "training_family_cv",
            "folds": folds,
            "seed": seed,
            "family_count": len(families),
            "selected_alpha": chosen["alpha"],
            "candidates": reports,
            "held_labels_used": False,
        }
        model._state["state_sha256"] = _digest(model._state)
        _validate(model._state)
        return model

    def predict(self, features, solvent_ids):
        provenance = _validate(self._state)
        provenance.authorize("predict")
        matrix = np.asarray(features, dtype=float)
        parameters = self._state["parameters"]
        solvents = tuple(solvent_ids)
        _require(
            matrix.ndim == 2 and matrix.shape == (len(solvents), parameters["raw_columns"]),
            "Query alignment",
        )
        _require(
            all(isinstance(s, str) and s.strip() for s in solvents), "Opaque solvent IDs required"
        )
        finite = np.isfinite(matrix).all(axis=1)
        outside = ~finite | np.any(
            (matrix < np.array(parameters["train_min"]) - OOD_TOLERANCE)
            | (matrix > np.array(parameters["train_max"]) + OOD_TOLERANCE),
            axis=1,
        )
        logits, probabilities = np.full(len(matrix), np.nan), np.full(len(matrix), np.nan)
        valid, reason = (
            np.zeros(len(matrix), bool),
            np.full(len(matrix), "nonfinite_query_features", dtype="U44"),
        )
        active = np.array(parameters["active"], bool)
        indices = np.flatnonzero(finite)
        with np.errstate(over="ignore", invalid="ignore"):
            z = (matrix[indices][:, active] - np.array(parameters["mean"])[active]) / np.array(
                parameters["scale"]
            )[active]
            design, supported = _design(
                z, [solvents[i] for i in indices], self._state["solvent_catalogue"]
            )
            eta = design @ np.array(self._state["weights"]) + self._state["intercept"]
        accepted = supported & np.isfinite(eta) & np.isfinite(design).all(axis=1)
        valid[indices] = accepted
        logits[indices[accepted]] = eta[accepted]
        probabilities[indices[accepted]] = expit(eta[accepted])
        reason[indices] = np.where(
            ~supported,
            "unsupported_solvent_coordinate",
            np.where(accepted, "", "nonfinite_logit_or_standardized_features"),
        )
        return {
            "logit": logits,
            "probability": probabilities,
            "valid": valid,
            "OOD": outside,
            "reason": reason,
            "model_state_sha256": self._state["state_sha256"],
            "provenance": copy.deepcopy(self._state["provenance"]),
            "calibrated_probability_or_coverage": False,
            "material_applicability_qualified": False,
            "physical_accuracy_qualified": False,
            "HSP_predictions": None,
        }

    def to_dict(self):
        _validate(self._state)
        return copy.deepcopy(self._state)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value):
        return cls(value)

    @classmethod
    def from_json(cls, text):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, "Duplicate model JSON key")
                result[key] = value
            return result

        return cls(json.loads(text, object_pairs_hook=unique))


def binary_compatibility_scores(labels, logits, rows):
    """Family/pair-balanced reference-label metrics; every supplied row is counted."""
    records = _rows(rows)
    y, eta = _labels(labels, len(records)), np.asarray(logits, dtype=float)
    _require(
        eta.shape == y.shape and np.isfinite(eta).all(), "Full finite metric population required"
    )
    weights, probability = family_pair_weights(records), expit(eta)
    class_scores = []
    for label in (0.0, 1.0):
        selected = y == label
        if selected.any():
            class_scores.append(
                float(
                    weights[selected]
                    @ ((probability[selected] >= 0.5) == label)
                    / np.sum(weights[selected])
                )
            )
    return {
        "Brier": float(weights @ ((probability - y) ** 2)),
        "logloss": float(weights @ (np.logaddexp(0.0, eta) - y * eta)),
        "balanced_accuracy": float(np.mean(class_scores)) if len(class_scores) == 2 else None,
        "rows": len(records),
        "families": len({r.family_id for r in records}),
        "pairs": len({(r.pair_id, r.solvent_id) for r in records}),
        "class_support": {str(label): int(np.sum(y == label)) for label in (0, 1)},
    }
