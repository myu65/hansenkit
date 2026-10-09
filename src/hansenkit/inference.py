"""Applicability is checked before features or numerical predictions are produced."""

import numpy as np

from .data import TARGETS
from .evaluation import domain_diagnostics
from .models import HSPModel
from .schema import INPUT_ADAPTER
from .scope import assess_scope


def predict_record(model: HSPModel, record: dict, encoder=None) -> dict:
    parsed = INPUT_ADAPTER.validate_python(record)
    base = {
        "label_kind": model.provenance["label_kind"],
        "units": model.provenance["units"],
        "real_accuracy_validated": False,
    }
    scope = assess_scope(parsed)
    if not scope.supported:
        return {
            **base,
            "status": "unsupported",
            "reasons": list(scope.reasons),
            "predictions": None,
        }
    diagnostics = domain_diagnostics(model, [parsed.smiles])
    ood, extrapolation = bool(diagnostics["ood"][0]), bool(diagnostics["extrapolation"][0])
    base.update(
        {
            "canonical_smiles": parsed.smiles,
            "nearest_training_tanimoto": float(diagnostics["nearest_training_tanimoto"][0]),
            "ood": ood,
            "extrapolation": extrapolation,
        }
    )
    if ood or extrapolation:
        return {
            **base,
            "status": "out_of_domain",
            "predictions": None,
            "reasons": [
                name for name, flag in (("ood", ood), ("extrapolation", extrapolation)) if flag
            ],
        }
    prediction = model.predict((parsed.smiles,), encoder)[0]
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        return {
            **base,
            "status": "invalid_model_output",
            "predictions": None,
            "reasons": ["nonphysical_or_nonfinite_output"],
        }
    output = {
        **base,
        "status": "synthetic_demo" if base["label_kind"] == "synthetic" else "research_only",
        "predictions": dict(zip(TARGETS, prediction.tolist(), strict=True)),
        "reasons": [],
        "intervals": None,
    }
    if model.conformal_radius is not None:
        output["intervals"] = {
            target: [
                float(prediction[i] - model.conformal_radius[i]),
                float(prediction[i] + model.conformal_radius[i]),
            ]
            for i, target in enumerate(TARGETS)
        }
    return output
