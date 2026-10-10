import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .chemistry import chemical_features
from .data import TARGETS, Dataset
from .encoders import MorganEncoder
from .models import HSPModel


def component_metrics(y, prediction):
    if len(y) == 0:
        return {"rows": 0, "components": None}
    output = {}
    for i, target in enumerate(TARGETS):
        truth, estimate = y[:, i], prediction[:, i]
        r2 = None
        if len(truth) >= 2 and np.var(truth) > 0:
            r2 = float(r2_score(truth, estimate))
        output[target] = {
            "mae": float(mean_absolute_error(truth, estimate)),
            "rmse": float(math.sqrt(mean_squared_error(truth, estimate))),
            "r2": r2,
        }
    return {"rows": len(y), "components": output}


def domain_diagnostics(model: HSPModel, smiles, similarity_threshold=0.35):
    encoder = MorganEncoder()
    ref, query = encoder.transform(model.train_smiles), encoder.transform(smiles)
    intersection = query @ ref.T
    union = query.sum(1)[:, None] + ref.sum(1)[None, :] - intersection
    similarity = np.max(intersection / np.maximum(union, 1), axis=1)
    features = np.stack([chemical_features(s) for s in smiles])
    extrapolation = (
        (features < model.feature_min - 1e-8) | (features > model.feature_max + 1e-8)
    ).any(1)
    ood = similarity < similarity_threshold
    return {
        "nearest_training_tanimoto": similarity,
        "ood": ood,
        "extrapolation": extrapolation,
        "threshold": similarity_threshold,
    }


def calibrate(model: HSPModel, dataset: Dataset, indices, groups, encoder=None, alpha=0.2):
    dataset.manifest.authorize("evaluate")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if dataset.manifest.label_kind != model.provenance["label_kind"]:
        raise ValueError("Calibration and training label kinds must match")
    smiles = tuple(dataset.smiles[i] for i in indices)
    if set(smiles) & set(model.train_smiles):
        raise ValueError("Calibration contains training molecules")
    prediction = model.predict(smiles, encoder)
    errors = np.abs(dataset.targets[indices] - prediction)
    cal_groups = groups[indices]
    # One score per family/component: maximum error within that calibration family.
    scores = np.stack([errors[cal_groups == g].max(0) for g in sorted(set(cal_groups))])
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    model.conformal_radius = np.sort(scores, axis=0)[rank - 1] if rank <= len(scores) else None
    model.calibration_info = {
        "method": "split-conformal-group-max-per-component",
        "alpha": alpha,
        "nominal_component_group_coverage": 1 - alpha,
        "calibration_rows": len(indices),
        "calibration_groups": len(scores),
        "calibration_label_kind": dataset.manifest.label_kind,
        "calibration_dataset_sha256": dataset.manifest.sha256,
        "finite_interval_available": model.conformal_radius is not None,
        "caveat": (
            "Coverage requires exchangeable groups; scaffold/OOD shifts can violate this. "
            "Not a physical confidence claim. Components are not jointly calibrated."
        ),
    }


def evaluate(model: HSPModel, dataset: Dataset, indices, groups, encoder=None):
    dataset.manifest.authorize("evaluate")
    if dataset.manifest.label_kind != model.provenance[
        "label_kind"
    ] and dataset.manifest.label_kind not in {"experimental", "published_reference"}:
        raise ValueError(
            "Use matching labels or a separately cleared independent experimental evaluation"
        )
    smiles = tuple(dataset.smiles[i] for i in indices)
    if set(smiles) & set(model.train_smiles):
        raise ValueError("Evaluation contains training molecules")
    prediction, truth = model.predict(smiles, encoder), dataset.targets[indices]
    domain = domain_diagnostics(model, smiles)
    uncertainty = {
        **(model.calibration_info or {}),
        "calibration_matches_evaluation_labels": (
            (model.calibration_info or {}).get("calibration_label_kind")
            == dataset.manifest.label_kind
        ),
        "empirical_row_coverage": None,
        "interval_width": None,
    }
    if model.conformal_radius is not None:
        covered = np.abs(truth - prediction) <= model.conformal_radius
        uncertainty["empirical_row_coverage"] = dict(
            zip(TARGETS, covered.mean(0).tolist(), strict=True)
        )
        uncertainty["interval_width"] = dict(
            zip(TARGETS, (2 * model.conformal_radius).tolist(), strict=True)
        )
        test_groups = groups[indices]
        group_cover = np.stack([covered[test_groups == g].all(0) for g in sorted(set(test_groups))])
        uncertainty["empirical_group_coverage"] = dict(
            zip(TARGETS, group_cover.mean(0).tolist(), strict=True)
        )
    subsets = {}
    for name in ("ood", "extrapolation"):
        mask = domain[name]
        subsets[name] = component_metrics(truth[mask], prediction[mask])
        subsets[f"non_{name}"] = component_metrics(truth[~mask], prediction[~mask])
    return {
        "label_kind": dataset.manifest.label_kind,
        "training_label_kind": model.provenance["label_kind"],
        "evaluation_basis": "independent_measurements"
        if dataset.manifest.label_kind == "experimental"
        else dataset.manifest.label_kind,
        "real_accuracy_validated": False,
        "metrics": component_metrics(truth, prediction),
        "subsets": subsets,
        "domain": {
            "ood_rows": int(domain["ood"].sum()),
            "extrapolation_rows": int(domain["extrapolation"].sum()),
            "similarity_threshold": domain["threshold"],
            "reference": "training-only Morgan fingerprint",
        },
        "uncertainty": uncertainty,
        "warning": (
            "Synthetic/teacher agreement is not experimental HSP accuracy; "
            "this PoC has no independent real-data validation."
        ),
    }
