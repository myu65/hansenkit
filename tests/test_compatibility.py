"""Project-authored array fixtures; no molecular dataset, source files or weights."""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from hansenkit import compatibility as c


def provenance(**changes):
    payload = {
        "label_kind": "synthetic_polymer_compatibility",
        "dataset_id": "authored-array-fixture",
        "source_sha256": hashlib.sha256(b"project-authored compatibility arrays").hexdigest(),
        "source": "Original deterministic synthetic arrays",
        "license": "MIT",
        "rights_status": "approved",
        "permission_evidence": "Fixture authored for this project",
        "approved_by": "Fixture author",
        "audit_basis": "documented_permission",
        "training_allowed": True,
        "derived_weights_allowed": True,
        "local_evaluation_allowed": True,
        "measurement_lineage": "authored",
        "input_representation": "authored_array_fixture",
    }
    payload.update(changes)
    return c.CompatibilityProvenance(**payload)


def arrays(families=12, columns=4):
    rng = np.random.default_rng(41685)
    x = np.repeat(rng.normal(size=(families, columns)), 2, axis=0)
    rows = [
        c.CompatibilityRow(f"own-{f}-{j}", f"family-{f}", f"pair-{f}", f"solvent-{j}")
        for f in range(families)
        for j in range(2)
    ]
    y = np.array([(f + j) % 2 for f in range(families) for j in range(2)], float)
    return x, rows, y


class PoisonArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Array converted before rights authorization")


@pytest.mark.parametrize(
    "changes",
    [
        {"rights_status": "pending"},
        {"rights_status": "denied"},
        {"training_allowed": False},
        {"derived_weights_allowed": False},
        {"allowed_role": "evaluation_only"},
    ],
)
def test_rights_before_arrays(changes):
    with pytest.raises(ValueError):
        c.CompatibilityClassifier.fit(
            PoisonArray(), [], PoisonArray(), provenance=provenance(**changes)
        )


def test_typed_binary_provenance_and_lineage():
    x, rows, y = arrays()
    with pytest.raises(ValueError, match="typed"):
        c.CompatibilityClassifier.fit(x, rows, y, provenance={})
    with pytest.raises(ValidationError):
        provenance(label_kind="experimental_polymer_compatibility")
    with pytest.raises(ValidationError):
        provenance(label_kind="published_reference_polymer_compatibility_auxiliary")
    reference = provenance(
        label_kind="published_reference_polymer_compatibility_auxiliary",
        measurement_lineage="unknown",
        input_representation="reviewed_source_unit_arrays",
        audit_basis="operator_assumption",
    )
    assert reference.use_scope == "scientific_private_noncommercial"
    with pytest.raises(ValidationError):
        provenance(HSP_predictions=[1, 2, 3])
    with pytest.raises(ValidationError):
        provenance(physical_accuracy_qualified=True)
    with pytest.raises(ValidationError):
        provenance(training_allowed=1)


@pytest.mark.parametrize(
    "field", ["dataset_id", "source", "license", "permission_evidence", "approved_by"]
)
def test_blank_source_evidence_refused_before_arrays(field):
    with pytest.raises(ValidationError, match="Nonblank"):
        c.CompatibilityClassifier.fit(
            PoisonArray(), [], PoisonArray(), provenance=provenance(**{field: " \t\n "})
        )
    literal = "  original source evidence  "
    assert getattr(provenance(**{field: literal}), field) == literal


def test_prediction_reference_provenance_snapshot_and_model_identity():
    x, rows, y = arrays()
    declared = provenance(
        label_kind="published_reference_polymer_compatibility_auxiliary",
        measurement_lineage="unknown",
        input_representation="reviewed_source_unit_arrays",
        audit_basis="operator_assumption",
    )
    model = c.CompatibilityClassifier.fit(x, rows, y, provenance=declared)
    state = model.to_dict()
    result = model.predict(x, [r.solvent_id for r in rows])
    assert result["model_state_sha256"] == state["state_sha256"]
    assert result["provenance"] == state["provenance"] == declared.model_dump(mode="json")
    assert result["provenance"]["label_kind"] == declared.label_kind
    assert result["provenance"]["use_scope"] == "scientific_private_noncommercial"
    assert result["provenance"]["source_sha256"] == declared.source_sha256
    assert result["provenance"]["public_redistribution"] is False
    assert result["provenance"]["commercial_allowed"] is False
    assert result["physical_accuracy_qualified"] is False and result["HSP_predictions"] is None
    result["provenance"]["source"] = "mutated result"
    result["provenance"]["physical_accuracy_qualified"] = True
    assert model.to_dict() == state
    assert model.predict(x, [r.solvent_id for r in rows])["provenance"] == state["provenance"]


def test_no_hsp_triple_labels_or_nonbinary_repair():
    x, rows, y = arrays()
    for invalid in (np.column_stack((y, y, y)), np.full(len(y), 0.5), np.full(len(y), np.nan)):
        with pytest.raises(ValueError, match="binary"):
            c.CompatibilityClassifier.fit(x, rows, invalid, provenance=provenance())


def test_conflicting_repeated_pair_weight_and_full_metrics():
    rows = [
        c.CompatibilityRow("r1", "f1", "p1", "s"),
        c.CompatibilityRow("r2", "f2", "p2", "s"),
        c.CompatibilityRow("r3", "f1", "p1", "s"),
    ]
    np.testing.assert_array_equal(c.family_pair_weights(rows), [0.25, 0.5, 0.25])
    score = c.binary_compatibility_scores([0, 1, 1], [0, 0, 0], rows)
    assert score["rows"] == 3 and score["families"] == 2 and score["pairs"] == 2
    assert score["Brier"] == 0.25
    assert c.binary_compatibility_scores([0, 0, 0], [0, 0, 0], rows)["balanced_accuracy"] is None


def test_pair_family_and_source_identity_gates():
    x, rows, y = arrays()
    bad = list(rows)
    bad[-1] = c.CompatibilityRow(rows[0].row_id, "other", "other", "solvent-0")
    with pytest.raises(ValueError, match="Unique"):
        c.CompatibilityClassifier.fit(x, bad, y, provenance=provenance())
    bad[-1] = c.CompatibilityRow("own-new", "other", rows[0].pair_id, "solvent-0")
    with pytest.raises(ValueError, match="families"):
        c.CompatibilityClassifier.fit(x, bad, y, provenance=provenance())


def test_gradient_independent_difference_extreme_zeros_preserved():
    rng = np.random.default_rng(1456)
    design = rng.normal(size=(11, 3))
    y = np.array([i % 2 for i in range(11)], float)
    weights, parameter = np.ones(11) / 11, np.array([0.3, -0.4, 0.2, 0.1])
    _, gradient = c.logistic_objective_gradient(parameter, design, y, weights, 0.2)
    for i in range(4):
        step = np.zeros(4)
        step[i] = 1e-6
        difference = (
            c.logistic_objective_gradient(parameter + step, design, y, weights, 0.2)[0]
            - c.logistic_objective_gradient(parameter - step, design, y, weights, 0.2)[0]
        ) / 2e-6
        assert gradient[i] == pytest.approx(difference, abs=2e-9)
    for intercept in (-1000.0, 1000.0):
        loss, gradient = c.logistic_objective_gradient(
            [0, 0, 0, intercept], design, y, weights, 1.0
        )
        assert np.isfinite(loss) and np.isfinite(gradient).all()


def test_train_scaling_query_poison_and_unsupported_coordinate():
    x, rows, y = arrays()
    x[:, -1] = 4.0
    model = c.CompatibilityClassifier.fit(x, rows, y, provenance=provenance())
    before = model.to_json()
    query = x[:3].copy()
    query[0, 0] = 1e90
    query[1, 0] = np.nan
    result = model.predict(query, ["solvent-0", "solvent-0", "unseen"])
    assert model.to_json() == before
    assert result["OOD"].tolist() == [True, True, False]
    assert result["valid"].tolist() == [True, False, False]
    assert np.isnan(result["probability"][1:]).all()
    assert result["reason"][2] == "unsupported_solvent_coordinate"
    assert not model.to_dict()["parameters"]["active"][-1]


def test_roundtrip_and_independent_forward():
    x, rows, y = arrays(columns=7)
    model = c.CompatibilityClassifier.fit(x, rows, y, provenance=provenance(), alpha=0.1)
    restored = c.CompatibilityClassifier.from_json(model.to_json())
    state = restored.to_dict()
    params = state["parameters"]
    active = np.array(params["active"], bool)
    standardized = (x[:, active] - np.array(params["mean"])[active]) / np.array(params["scale"])[
        active
    ]
    categorical = np.array(
        [[r.solvent_id == s for s in state["solvent_catalogue"]] for r in rows], float
    )
    expected = (
        np.column_stack((standardized, categorical)) @ np.array(state["weights"])
        + state["intercept"]
    )
    result = restored.predict(x, [r.solvent_id for r in rows])
    np.testing.assert_array_equal(result["logit"], expected)
    assert restored.to_dict()["provenance"]["source_sha256"] == provenance().source_sha256
    assert result["HSP_predictions"] is None
    state["weights"][0] += 1.0
    with pytest.raises(ValueError, match="integrity"):
        c.CompatibilityClassifier.from_dict(state)


@pytest.mark.parametrize(
    "change", ["rights", "convergence", "dimensions", "qualification", "runtime"]
)
def test_invalid_resealed_state_refuses(change):
    x, rows, y = arrays()
    state = c.CompatibilityClassifier.fit(x, rows, y, provenance=provenance()).to_dict()
    if change == "rights":
        state["provenance"]["training_allowed"] = False
    elif change == "convergence":
        state["diagnostics"]["gradient_inf"] = 0.1
    elif change == "dimensions":
        state["parameters"]["mean"] = [[value] for value in state["parameters"]["mean"]]
    elif change == "qualification":
        state["material_applicability_qualified"] = True
    else:
        state["runtime"]["numpy"] = "wrong"
    state["state_sha256"] = c._digest(state)
    with pytest.raises(ValueError):
        c.CompatibilityClassifier.from_dict(state)


def test_duplicate_json_keys_and_missing_encoder_vectors():
    with pytest.raises(ValueError, match="Duplicate"):
        c.CompatibilityClassifier.from_json('{"schema_version":1,"schema_version":1}')
    x, rows, y = arrays()
    x[0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        c.CompatibilityClassifier.fit(x, rows, y, provenance=provenance())


def test_single_class_and_low_coordinate_support_refusal():
    x, rows, y = arrays()
    with pytest.raises(c.CompatibilityFitRefusal, match="single_class"):
        c.CompatibilityClassifier.fit(x, rows, np.zeros(len(y)), provenance=provenance())
    bad = list(rows)
    bad[0] = c.CompatibilityRow(
        "own-special", rows[0].family_id, rows[0].pair_id, "one-family-only"
    )
    with pytest.raises(ValueError, match="two training"):
        c.CompatibilityClassifier.fit(x, bad, y, provenance=provenance())


def test_optimizer_failure_evidence_no_retry():
    x, rows, y = arrays()
    with patch.object(c, "minimize", side_effect=RuntimeError("owned failure")) as call:
        with pytest.raises(c.CompatibilityFitRefusal) as failure:
            c.CompatibilityClassifier.fit(x, rows, y, provenance=provenance())
    assert call.call_count == 1
    assert failure.value.diagnostics["exception_type"] == "RuntimeError"

    def fail(function, initial, **kwargs):
        value, gradient = function(initial, *kwargs["args"])
        return SimpleNamespace(
            x=initial,
            fun=value,
            success=False,
            status=2,
            message="owned failure",
            nit=1,
            jac=gradient,
        )

    with patch.object(c, "minimize", side_effect=fail):
        with pytest.raises(c.CompatibilityFitRefusal) as failure:
            c.CompatibilityClassifier.fit(x, rows, y, provenance=provenance())
    assert failure.value.diagnostics["status"] == 2


def test_training_family_cv_fold_preprocessing_and_held_poison():
    x, rows, y = arrays(families=15, columns=3)
    fitted = []
    original = c._scale_fit

    def watched(matrix):
        fitted.append(matrix.copy())
        return original(matrix)

    with patch.object(c, "_scale_fit", side_effect=watched):
        model = c.CompatibilityClassifier.select(x, rows, y, provenance=provenance())
    assert len(fitted) == 16  # Three alphas x five training folds plus final training-only fit.
    assert all(len(matrix) < len(x) for matrix in fitted[:-1])
    assert len(fitted[-1]) == len(x)
    first = model.to_json()
    model.predict(np.full((2, 3), 1e99), ["solvent-0", "solvent-1"])
    assert model.to_json() == first
    replay = c.CompatibilityClassifier.select(x.copy(), rows, y.copy(), provenance=provenance())
    assert replay.to_json() == first
    assert not model.to_dict()["selection"]["held_labels_used"]
    damaged = model.to_dict()
    damaged["selection"]["held_labels_used"] = True
    damaged["state_sha256"] = c._digest(damaged)
    with pytest.raises(ValueError, match="training-only"):
        c.CompatibilityClassifier.from_dict(damaged)


def test_all_cv_failures_are_inadmissible_with_no_filtering():
    x, rows, y = arrays()
    with patch.object(
        c, "_fit", side_effect=c.CompatibilityFitRefusal("owned", {"gate": "refused"})
    ):
        with pytest.raises(c.CompatibilityFitRefusal) as failure:
            c.CompatibilityClassifier.select(x, rows, y, provenance=provenance())
    report = failure.value.diagnostics["candidates"]
    assert len(report) == 3 and all(r["inadmissible"] for r in report)
    assert sum(len(r["failures"]) for r in report) == 15


def test_prediction_rights_before_query_conversion():
    x, rows, y = arrays()
    model = c.CompatibilityClassifier.fit(
        x, rows, y, provenance=provenance(local_evaluation_allowed=False)
    )
    with pytest.raises(ValueError, match="evaluation"):
        model.predict(PoisonArray(), [])
