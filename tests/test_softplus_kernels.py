"""Project-owned numerical fixtures; no molecules, asset tables, encoder or MD."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import hansenkit.softplus_kernels as m


@pytest.fixture(autouse=True)
def cpu_threads():
    with threadpool_limits(limits=4):
        yield


def provenance():
    return {
        "label_kind": "synthetic",
        "training_allowed": True,
        "source": "original authored arithmetic fixtures",
        "permission_evidence": "Invented arrays owned by project; no scientific targets or assets",
        "approved_by": "fixture author",
        "audit_basis": "documented_permission",
        "use_scope": "scientific_private_noncommercial",
        "public_redistribution": False,
    }


def population():
    signs = np.repeat([-1.0, 1.0], 6)
    features = {
        "chemical": np.zeros((12, 105)),
        "embedding": np.zeros((12, 768)),
        "physics": np.full((12, 30), np.nan),
    }
    for block in features.values():
        block[:, 0] = signs
    # Quantum-like slots are arbitrary numbers and explicitly remain missing/masked.
    features["physics"][::3, 4] = 2.0
    target = np.column_stack((20 + 0.8 * signs, 8 + 2 * signs, 5 + 0.5 * signs))
    return features, target


@pytest.fixture(scope="module")
def trained():
    features, target = population()
    with threadpool_limits(limits=4):
        head = m.SoftplusKernelHead.fit(
            features, target, gamma=0.1, alpha=1.0, provenance=provenance()
        )
    return features, target, head


def test_original_objective_gradient_signed_penalty_and_chain_rule():
    rng = np.random.default_rng(612)
    factor, y, raw = rng.normal(size=(7, 3)), rng.uniform(0, 40, (7, 3)), rng.normal(size=12)
    y[0, 1:] = 0
    objective, gradient = m.objective_gradient(raw, factor, y, 0.7)
    expected_gradient = []
    for index in range(len(raw)):
        step = np.eye(len(raw))[index] * 1e-5
        hi = m.objective_gradient(raw + step, factor, y, 0.7)[0]
        lo = m.objective_gradient(raw - step, factor, y, 0.7)[0]
        expected_gradient.append((hi - lo) / 2e-5)
    np.testing.assert_allclose(gradient, expected_gradient, atol=3e-8, rtol=2e-7)
    values = np.array([0.001, 1.0, 80.0])
    scale = m.parameter_scales(values, 0.7, 7)
    mapped_loss, mapped_gradient = m.preconditioned_objective_gradient(
        raw * scale, factor, y, 0.7, values
    )
    np.testing.assert_allclose(mapped_loss, objective, atol=1e-12, rtol=1e-14)
    np.testing.assert_allclose(mapped_gradient * scale, gradient, atol=1e-13, rtol=1e-13)
    other = m.objective_gradient(raw, factor, y, 1.7)[1]
    np.testing.assert_allclose((other - gradient)[:-3], raw[:-3], atol=1e-13)
    np.testing.assert_array_equal(other[-3:], gradient[-3:])


def test_zero_targets_limiting_intercept_is_not_exact_global_zero():
    factor, y, raw = np.ones((5, 2)), np.zeros((5, 3)), np.zeros(9)
    first = m.objective_gradient(raw, factor, y, 1.0)[0]
    raw[-3:] = -5.0
    second = m.objective_gradient(raw, factor, y, 1.0)[0]
    assert 0 < second < first
    np.testing.assert_array_equal(m.original_targets(y, 5), y)
    values, valid, reasons = m.response(
        np.array([-1000.0, -745.0, 0.0, 1000.0, 1e308, np.nan, np.inf])
    )
    assert values[0] == 0 and values[2] == np.log(2.0) and values[3] == 1000.0
    assert valid[:5].all() and not valid[5:].any() and np.isnan(values[5:]).all()
    assert np.all(reasons[5:] == "nonfinite_latent")


def test_authored_all_zero_fit_preserves_limiting_solution_scope():
    features, target = population()
    target[:] = 0.0
    head = m.SoftplusKernelHead.fit(features, target, gamma=0.1, alpha=1.0, provenance=provenance())
    prediction, valid, reasons = head.predict(features)
    assert valid.all() and np.isfinite(prediction).all() and np.all(prediction >= 0)
    assert np.all(reasons == "") and head.state()["zero_training_targets_preserved"] is True
    assert (
        head.state()["optimizer_diagnostics_before_admissibility"][
            "local_convergence_only_no_global_minimum_claim"
        ]
        is True
    )
    assert head.state()["qualified_HSP_backend"] is False


def test_native_signed_kernel_forward_and_json_roundtrip(trained):
    features, target, head = trained
    state = head.state()
    expected_features = m._transform(features, state["preparation"])
    expected_kernel = np.zeros((12, 12))
    for name, kind, weight in (
        ("chemical", "linear", 0.4),
        ("embedding", "rbf", 0.4),
        ("physics", "rbf", 0.2),
    ):
        q = expected_features[name]
        if not state["preparation"]["blocks"][name]["active"]:
            continue
        if kind == "linear":
            term = q @ q.T / np.mean(np.sum(q * q, axis=1))
        else:
            term = np.exp(-0.1 * np.sum((q[:, None] - q[None]) ** 2, axis=2) / q.shape[1])
        expected_kernel += weight * term
    vectors, values = (
        np.asarray(state["eigenvectors_retained"]),
        np.asarray(state["eigenvalues_retained"]),
    )
    factor = vectors * np.sqrt(values)
    np.testing.assert_allclose(factor @ factor.T, expected_kernel, atol=1e-12, rtol=1e-12)
    eta = (expected_kernel @ (vectors / np.sqrt(values))) @ state["theta_signed"] + state[
        "intercept_signed_unpenalized"
    ]
    predicted, valid, reasons = head.predict(features)
    np.testing.assert_allclose(predicted, np.logaddexp(0.0, eta), atol=1e-12, rtol=1e-12)
    assert valid.all() and np.all(reasons == "") and (predicted >= 0).all()
    raw = np.r_[np.asarray(state["theta_signed"]).ravel(), state["intercept_signed_unpenalized"]]
    loss, gradient = m.objective_gradient(raw, factor, target, 1.0)
    cert = state["optimizer_diagnostics_before_admissibility"]
    assert cert["success"] and cert["status"] == 0 and np.max(np.abs(gradient)) <= 1e-5
    np.testing.assert_array_equal(gradient, cert["original_gradient"])
    assert loss == cert["final_objective"]
    restored = m.SoftplusKernelHead.restore(json.loads(json.dumps(state, allow_nan=False)))
    for actual, expected in zip(restored.predict(features), head.predict(features), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert restored.state()["provenance"] == provenance()
    assert state["qualified_HSP_backend"] is False and state["calibrated_HSP_uncertainty"] is False
    assert state["array_level_only"] is True and "training_y" not in state
    assert cert["local_convergence_only_no_global_minimum_claim"]


def test_full_768_input_and_unused_pca_contract(trained):
    features, _, head = trained
    query = {k: x[:1].copy() for k, x in features.items()}
    baseline = head.predict(query)[0]
    query["embedding"][0, 767] = 100.0
    changed = head.predict(query)[0]
    assert np.max(np.abs(changed - baseline)) > 1e-4
    state = head.state()
    state["preparation"]["pca"]["components"] = np.zeros_like(
        state["preparation"]["pca"]["components"]
    ).tolist()
    state["state_sha256"] = m._digest(state)
    revised = m.SoftplusKernelHead.restore(state)
    np.testing.assert_array_equal(revised.predict(query)[0], changed)


def test_external_features_and_targets_cannot_change_training_state(trained):
    features, target, head = trained
    before = head.state()
    held = {k: np.vstack((x, x[:3])) for k, x in features.items()}
    held_y = np.vstack((target, target[:3]))
    held["chemical"][12:] = 1e6
    held["embedding"][12:] = -1e6
    held["physics"][12:] = 1e6
    held_y[12:] = 1e7
    repeated = m.SoftplusKernelHead.fit(
        {k: x[:12] for k, x in held.items()},
        held_y[:12],
        gamma=0.1,
        alpha=1.0,
        provenance=provenance(),
    )
    assert repeated.state() == before
    repeated.predict({k: x[12:] for k, x in held.items()})
    assert repeated.state() == before
    state = head.state()
    state["provenance"]["source"] = "changed returned copy"
    assert head.state() == before


@pytest.mark.parametrize(
    "gram, message",
    [
        (np.diag([2.0, 1.0, -1e-5]), "PSD"),
        (np.zeros((3, 3)), "below"),
        (np.array([[1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "asymmetry"),
    ],
)
def test_spectral_material_defects_refuse_without_repair(gram, message):
    with pytest.raises(ValueError, match=message):
        m.spectral_basis(gram)


def test_rank_deficient_psd_roundoff_and_direct_ridge_initialization():
    factor = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, -1.0]])
    gram = factor @ factor.T
    basis, values, vectors, projection, diag = m.spectral_basis(gram)
    assert diag["retained_rank"] == 2 and diag["labels_used_for_eigenbasis"] is False
    np.testing.assert_allclose(gram @ projection, basis, atol=2e-13)
    y = np.array([[2.0, 3.0, 4.0], [0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    raw = m.ridge_initialization(y, values, vectors, 1.0)
    expected = gram @ np.linalg.solve(gram + np.eye(4), y - y.mean(0)) + y.mean(0)
    np.testing.assert_allclose(basis @ raw[:-3].reshape(2, 3) + raw[-3:], expected, atol=2e-13)
    _, retained, _, _, small = m.spectral_basis(np.diag([2.0, 1.0, -1e-10]))
    assert len(retained) == 2 and small["negative_roundoff_eigenvalue_count"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "not_approved",
        "evaluation_only",
        "mixed_labels",
        "empty_evidence",
        "redistribution",
        "commercial",
        "physical",
        "uncertainty",
        "missing_source",
        "nan_provenance",
    ],
)
def test_rights_and_single_label_kind_refuse_before_array_fit(monkeypatch, mutation):
    p = provenance()
    if mutation == "not_approved":
        p["training_allowed"] = False
    elif mutation == "evaluation_only":
        p["evaluation_only"] = True
    elif mutation == "mixed_labels":
        p["label_kind"] = ["synthetic", "experimental"]
    elif mutation == "empty_evidence":
        p["permission_evidence"] = " "
    elif mutation == "redistribution":
        p["public_redistribution"] = True
    elif mutation == "commercial":
        p["commercial_allowed"] = True
    elif mutation == "physical":
        p["qualified_HSP_backend"] = True
    elif mutation == "uncertainty":
        p["calibrated_HSP_uncertainty"] = True
    elif mutation == "missing_source":
        del p["source"]
    else:
        p["unapproved_number"] = np.nan

    def poison(*args, **kwargs):
        raise AssertionError("Training arrays must not be fitted after rights refusal")

    monkeypatch.setattr(m.BlockKernelPreparation, "fit", poison)
    with pytest.raises(ValueError):
        m.SoftplusKernelHead.fit({}, [], gamma=0.1, alpha=1.0, provenance=p)


@pytest.mark.parametrize(
    "gamma, alpha", [(2.0, 1.0), (0.1, 0.001), (True, 1.0), (0.1, True), (0.1, np.nan)]
)
def test_configuration_refuses_before_solver(monkeypatch, gamma, alpha):
    monkeypatch.setattr(
        m, "minimize", lambda *args, **kwargs: pytest.fail("Solver must not execute")
    )
    with pytest.raises(ValueError):
        m.SoftplusKernelHead.fit(*population(), gamma=gamma, alpha=alpha, provenance=provenance())


@pytest.mark.parametrize(
    "mutation", ["negative", "nonfinite", "shape", "feature_shape", "feature_inf"]
)
def test_bad_targets_and_features_refuse(monkeypatch, mutation):
    features, target = population()
    if mutation == "negative":
        target[0, 0] = -1.0
    elif mutation == "nonfinite":
        target[0, 0] = np.inf
    elif mutation == "shape":
        target = target[:-1]
    elif mutation == "feature_shape":
        features["embedding"] = features["embedding"][:, :-1]
    else:
        features["chemical"][0, 0] = np.inf
    monkeypatch.setattr(
        m, "minimize", lambda *args, **kwargs: pytest.fail("Solver must not execute")
    )
    with pytest.raises(ValueError):
        m.SoftplusKernelHead.fit(features, target, gamma=0.1, alpha=1.0, provenance=provenance())


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "provenance",
        "schema_bool",
        "shape",
        "projection",
        "trace",
        "runtime",
        "status",
        "status_bool",
        "gradient",
        "chain",
        "objective",
        "missing",
        "scalar",
        "physical",
    ],
)
def test_malformed_json_state_refuses_even_with_recomputed_digest(trained, mutation):
    features, _, head = trained
    state = head.state()
    cert = state["optimizer_diagnostics_before_admissibility"]
    if mutation == "hash":
        state["theta_signed"][0][0] += 1.0
    elif mutation == "provenance":
        state["provenance"]["training_allowed"] = False
    elif mutation == "schema_bool":
        state["schema_version"] = True
    elif mutation == "shape":
        state["theta_signed"] = []
    elif mutation == "projection":
        state["query_projection"][0][0] += 1.0
    elif mutation == "trace":
        state["preparation"]["blocks"]["embedding"]["coordinate_trace"] *= 2
    elif mutation == "runtime":
        state["runtime_versions"]["scipy"] = "unknown"
    elif mutation == "status":
        cert["status"] = 2
    elif mutation == "status_bool":
        cert["status"] = False
    elif mutation == "gradient":
        cert["original_gradient"][0] = 1.0
    elif mutation == "chain":
        cert["optimizer_gradient"][0] += 0.1
    elif mutation == "objective":
        cert["final_objective"] = cert["initial_objective"] + 1.0
    elif mutation == "missing":
        del state["preparation"]
    elif mutation == "scalar":
        state["preparation"]["blocks"]["chemical"]["training"] = 0.0
    else:
        state["qualified_HSP_backend"] = True
    if mutation != "hash":
        state["state_sha256"] = m._digest(state)
    with pytest.raises(ValueError):
        m.SoftplusKernelHead.restore(state)
    corrupted = m.SoftplusKernelHead(state)
    with pytest.raises(ValueError):
        corrupted.predict(features)


def test_native_failure_remains_refused_even_with_passing_original_gradient(monkeypatch, trained):
    features, target, head = trained
    state = head.state()
    raw = np.r_[np.asarray(state["theta_signed"]).ravel(), state["intercept_signed_unpenalized"]]
    mapped = raw * m.parameter_scales(state["eigenvalues_retained"], 1.0, 12)
    calls = []

    def failed(callback, initial, **kwargs):
        calls.append(True)
        fun, jac = callback(mapped)
        return SimpleNamespace(
            success=False,
            status=2,
            message="authored native status2",
            x=mapped,
            fun=fun,
            jac=jac,
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr(m, "minimize", failed)
    with pytest.raises(m.NumericalRefusal, match="did not converge") as caught:
        m.SoftplusKernelHead.fit(features, target, gamma=0.1, alpha=1.0, provenance=provenance())
    diag = caught.value.diagnostics
    assert len(calls) == 1 and diag["original_gradient_gate_pass"] and diag["status"] == 2
    assert diag["original_gradient"] and diag["optimizer_gradient"] and not diag["success"]
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize("mutation", ["vectors", "values"])
def test_orthonormal_redigested_wrong_eigensystem_refuses_without_eigenfit(
    monkeypatch, trained, mutation
):
    _, _, head = trained
    state = head.state()
    values = np.asarray(state["eigenvalues_retained"])
    vectors = np.asarray(state["eigenvectors_retained"])
    if mutation == "vectors":
        vectors = np.roll(vectors, 1, axis=0)
    else:
        values *= 1.01
    state["eigenvalues_retained"] = values.tolist()
    state["eigenvectors_retained"] = vectors.tolist()
    state["query_projection"] = (vectors / np.sqrt(values)).tolist()
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=1e-12)
    state["state_sha256"] = m._digest(state)
    monkeypatch.setattr(
        np.linalg, "eigh", lambda *args, **kwargs: pytest.fail("Reload cannot refit eigenbasis")
    )
    with pytest.raises(ValueError, match="spectral basis"):
        m.SoftplusKernelHead.restore(state)


def test_claimed_success_consistent_loss_cannot_bypass_raw_gradient(monkeypatch):
    def claimed(callback, initial, **kwargs):
        fun, jac = callback(initial)
        return SimpleNamespace(
            success=True,
            status=0,
            message="authored nonstationary success",
            x=initial,
            fun=fun,
            jac=jac,
            nit=0,
            nfev=1,
        )

    monkeypatch.setattr(m, "minimize", claimed)
    with pytest.raises(m.NumericalRefusal, match="gradient/objective") as caught:
        m.SoftplusKernelHead.fit(*population(), gamma=0.1, alpha=1.0, provenance=provenance())
    diag = caught.value.diagnostics
    assert diag["objective_consistent"] and diag["optimizer_jacobian_consistent"]
    assert not diag["original_gradient_gate_pass"]


def test_optimizer_exception_and_nonfinite_result_keep_json_safe_evidence(monkeypatch):
    def raised(callback, initial, **kwargs):
        callback(initial)
        raise ValueError("authored optimizer error")

    monkeypatch.setattr(m, "minimize", raised)
    with pytest.raises(m.NumericalRefusal, match="no retry") as caught:
        m.SoftplusKernelHead.fit(*population(), gamma=0.1, alpha=1.0, provenance=provenance())
    diag = caught.value.diagnostics
    assert diag["result_returned"] is False and diag["status"] is None
    assert diag["last_finite_original_gradient"] and diag["last_finite_optimizer_gradient"]
    json.dumps(diag, allow_nan=False)

    def nonfinite(callback, initial, **kwargs):
        return SimpleNamespace(
            success=True,
            status=0,
            x=np.full_like(initial, np.nan),
            fun=np.inf,
            jac=np.full_like(initial, np.inf),
        )

    monkeypatch.setattr(m, "minimize", nonfinite)
    with pytest.raises(m.NumericalRefusal, match="Nonfinite") as caught:
        m.SoftplusKernelHead.fit(*population(), gamma=0.1, alpha=1.0, provenance=provenance())
    diag = caught.value.diagnostics
    assert diag["result_fun"] is None and diag["original_gradient"] is None
    assert all(value is None for value in diag["reported_optimizer_jacobian"])
    json.dumps(diag, allow_nan=False)


def test_loss_nonincrease_gate_has_no_retry_or_fallback(monkeypatch, trained):
    features, target, head = trained
    cert = head.state()["optimizer_diagnostics_before_admissibility"]
    cert["objective_not_increased"] = False
    original = np.r_[
        np.asarray(head.state()["theta_signed"]).ravel(),
        head.state()["intercept_signed_unpenalized"],
    ]
    calls = []

    def result(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace()

    monkeypatch.setattr(m, "minimize", result)
    monkeypatch.setattr(m, "optimizer_diagnostics", lambda *args: (cert, original))
    with pytest.raises(m.NumericalRefusal, match="exceeds"):
        m.SoftplusKernelHead.fit(features, target, gamma=0.1, alpha=1.0, provenance=provenance())
    assert calls == [True]


def test_fit_predict_never_read_assets_or_execute_encoder(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("No asset reads permitted"))
    features, target = population()
    head = m.SoftplusKernelHead.fit(features, target, gamma=0.1, alpha=1.0, provenance=provenance())
    predicted, valid, _ = head.predict(features)
    assert valid.all() and np.isfinite(predicted).all()


def test_finite_signed_head_overflow_returns_refusal_mask_without_repair(trained):
    features, _, head = trained
    state = head.state()
    state["theta_signed"] = np.full_like(state["theta_signed"], 1e308).tolist()
    state["state_sha256"] = m._digest(state)
    diagnostic = m.SoftplusKernelHead.restore(state)
    query = {key: values[:1].copy() for key, values in features.items()}
    query["chemical"][0, 0] = 1e50
    prediction, valid, reason = diagnostic.predict(query)
    assert not valid.any() and np.isnan(prediction).all()
    assert np.all(reason == "nonfinite_latent")
