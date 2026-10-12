"""Array-only learned softplus kernels; no chemical, rights or accuracy qualification.

Callers review assets and isolate families before supplying training arrays. This
module selects no configuration, reads no files and executes no encoder or MD.
Its unpenalized intercept has only a limiting optimum for an all-zero target
column; local numerical convergence does not guarantee a global minimum.
"""

import copy
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .block_kernels import BlockKernelPreparation, _gram, _matrix, _transform

DELTA_UNIT_MPA_HALF = 1.0
PSD_RELATIVE_TOLERANCE = 1e-9
SYMMETRY_RELATIVE_TOLERANCE = 1e-12
EIGEN_CUTOFF_RELATIVE = 1e-12
FINAL_GRADIENT_INFINITY_GATE = 1e-5
OPTIMIZER_OPTIONS = {
    "maxiter": 3000,
    "maxfun": 10000,
    "maxls": 40,
    "maxcor": 10,
    "ftol": 0.0,
    "gtol": 1e-9,
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def original_targets(value, rows):
    y = np.asarray(value, float)
    require(
        y.shape == (rows, 3) and np.isfinite(y).all() and np.all(y >= 0) and np.all(y <= 1e100),
        "Finite nonnegative original delta targets required",
    )
    return y / DELTA_UNIT_MPA_HALF


def response(latent):
    eta = np.asarray(latent, float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = DELTA_UNIT_MPA_HALF * np.logaddexp(0.0, eta)
    finite_latent, finite_output = np.isfinite(eta), np.isfinite(value)
    valid = finite_latent & finite_output & (value >= 0)
    reason = np.full(value.shape, "", dtype="U40")
    reason[~finite_latent] = "nonfinite_latent"
    reason[finite_latent & ~finite_output] = "nonfinite_softplus_response"
    return np.where(valid, value, np.nan), valid, reason


def objective_gradient(parameter, factor, y, alpha):
    factor, y = np.asarray(factor, float), np.asarray(y, float)
    rows, rank = factor.shape
    vector = np.asarray(parameter, float)
    require(
        vector.shape == (3 * (rank + 1),)
        and y.shape == (rows, 3)
        and np.isfinite(vector).all()
        and np.isfinite(factor).all()
        and np.isfinite(y).all()
        and np.isfinite(alpha)
        and alpha > 0,
        "Invalid objective inputs",
    )
    theta, intercept = vector[: 3 * rank].reshape(rank, 3), vector[3 * rank :]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        eta = factor @ theta + intercept
        predicted = np.logaddexp(0.0, eta)
        derivative = np.exp(-np.logaddexp(0.0, -eta))
        error = predicted - y
        weighted = error * derivative
        loss = 0.5 * np.sum(error * error) + 0.5 * alpha * np.sum(theta * theta)
        gradient = np.r_[(factor.T @ weighted + alpha * theta).ravel(), weighted.sum(0)]
    require(
        np.isfinite(eta).all() and np.isfinite(loss) and np.isfinite(gradient).all(),
        "Nonfinite softplus objective/gradient",
    )
    return float(loss), gradient


def parameter_scales(eigenvalues, alpha, rows):
    values = np.asarray(eigenvalues, float)
    require(
        values.ndim == 1
        and len(values) > 0
        and np.isfinite(values).all()
        and np.all(values > 0)
        and np.isfinite(alpha)
        and alpha > 0
        and type(rows) is int
        and rows >= 3,
        "Invalid same-objective preconditioner",
    )
    scale = np.r_[np.repeat(np.sqrt(values + alpha), 3), np.full(3, np.sqrt(rows))]
    require(np.isfinite(scale).all() and np.all(scale > 0), "Preconditioner overflow")
    return scale


def preconditioned_objective_gradient(parameter, factor, y, alpha, eigenvalues):
    scale = parameter_scales(eigenvalues, alpha, len(factor))
    vector = np.asarray(parameter, float)
    require(
        vector.shape == scale.shape and np.isfinite(vector).all(), "Invalid optimizer coordinates"
    )
    original = vector / scale
    loss, original_gradient = objective_gradient(original, factor, y, alpha)
    gradient = original_gradient / scale
    require(np.isfinite(gradient).all(), "Preconditioned chain-rule gradient overflow")
    return loss, gradient


class NumericalRefusal(ValueError):
    """Per-fit optimizer evidence survives every numerical refusal."""

    def __init__(self, reason, diagnostics):
        self.reason = reason
        self.diagnostics = copy.deepcopy(diagnostics)
        super().__init__(reason)


def optimizer_diagnostics(result, factor, target, alpha, eigenvalues, initial_loss):
    """Record status and both gradients before checking admissibility; JSON finite/null."""

    def scalar(value):
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        return number if np.isfinite(number) else None

    def infinity_norm(value):
        array = np.asarray(value, float)
        return scalar(np.max(np.abs(array))) if array.size else None

    scale = parameter_scales(eigenvalues, alpha, len(factor))
    diagnostic = {
        "method": "L-BFGS-B",
        "numerical_training_version": 2,
        "parameterization": (
            "theta=w/sqrt(lambda+alpha), b=c/sqrt(n); same original objective and prediction state"
        ),
        "options": copy.deepcopy(OPTIMIZER_OPTIONS),
        "result_returned": True,
        "success": bool(getattr(result, "success", False)),
        "status": getattr(result, "status", None),
        "message": str(getattr(result, "message", "")),
        "result_fun": scalar(getattr(result, "fun", None)),
        "iterations": getattr(result, "nit", None),
        "function_evaluations": getattr(result, "nfev", None),
        "initial_objective": float(initial_loss),
        "gradient_gate_original_coordinates": FINAL_GRADIENT_INFINITY_GATE,
        "original_gradient_infinity_norm": None,
        "optimizer_gradient_infinity_norm": None,
        "reported_optimizer_jacobian_infinity_norm": None,
        "original_gradient": None,
        "optimizer_gradient": None,
        "reported_optimizer_jacobian": None,
        "result_coordinates_finite_and_aligned": False,
        "original_objective_gradient_finite": False,
        "objective_consistent": False,
        "optimizer_jacobian_consistent": False,
        "original_gradient_gate_pass": False,
        "objective_not_increased": False,
        "local_convergence_only_no_global_minimum_claim": True,
    }
    for name in ("status", "iterations", "function_evaluations"):
        value = diagnostic[name]
        if isinstance(value, (int, np.integer)):
            diagnostic[name] = int(value)
        elif value is not None:
            diagnostic[name] = str(value)
    x = np.asarray(getattr(result, "x", []), float)
    jac = np.asarray(getattr(result, "jac", []), float)
    diagnostic["reported_optimizer_jacobian_infinity_norm"] = infinity_norm(jac)
    diagnostic["reported_optimizer_jacobian"] = [
        float(v) if np.isfinite(v) else None for v in jac.ravel()
    ]
    diagnostic["result_coordinates_finite_and_aligned"] = bool(
        x.shape == scale.shape and np.isfinite(x).all()
    )
    original = x / scale if x.shape == scale.shape else None
    if diagnostic["result_coordinates_finite_and_aligned"]:
        try:
            loss, gradient = objective_gradient(original, factor, target, alpha)
            opt_gradient = gradient / scale
            diagnostic.update(
                final_objective=loss,
                original_gradient_infinity_norm=infinity_norm(gradient),
                optimizer_gradient_infinity_norm=infinity_norm(opt_gradient),
                original_objective_gradient_finite=True,
                original_gradient=gradient.tolist(),
                optimizer_gradient=opt_gradient.tolist(),
                objective_consistent=bool(
                    np.allclose(getattr(result, "fun", np.nan), loss, rtol=1e-12, atol=1e-12)
                ),
                optimizer_jacobian_consistent=bool(
                    jac.shape == opt_gradient.shape
                    and np.isfinite(jac).all()
                    and np.allclose(jac, opt_gradient, rtol=1e-12, atol=1e-12)
                ),
                original_gradient_gate_pass=bool(
                    np.max(np.abs(gradient)) <= FINAL_GRADIENT_INFINITY_GATE
                ),
                objective_not_increased=bool(
                    loss <= initial_loss + 1e-10 * max(1.0, abs(initial_loss))
                ),
            )
        except (ValueError, TypeError, RuntimeError, FloatingPointError) as error:
            diagnostic["original_recomputation_error"] = str(error)
    return diagnostic, original


def spectral_basis(gram):
    gram = np.asarray(gram, float)
    require(
        gram.ndim == 2
        and gram.shape[0] == gram.shape[1]
        and len(gram) >= 3
        and np.isfinite(gram).all(),
        "Finite square training Gram required",
    )
    amplitude = max(1.0, float(np.max(np.abs(gram))))
    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    require(
        symmetry_error <= SYMMETRY_RELATIVE_TOLERANCE * amplitude,
        "Training Gram asymmetry exceeds roundoff policy",
    )
    symmetric = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(eigenvalues.max()))
    require(
        eigenvalues.min() >= -PSD_RELATIVE_TOLERANCE * scale,
        "Original additive Gram PSD gate failed",
    )
    cutoff = EIGEN_CUTOFF_RELATIVE * scale
    retained = eigenvalues > cutoff
    require(retained.any(), "All training kernel eigen-directions below frozen cutoff")
    positive, vectors = eigenvalues[retained], eigenvectors[:, retained]
    factor = vectors * np.sqrt(positive)
    projection = vectors / np.sqrt(positive)
    require(
        np.isfinite(factor).all() and np.isfinite(projection).all(), "Spectral coordinate overflow"
    )
    reconstruction = factor @ factor.T
    diagnostics = {
        "training_rows": len(gram),
        "retained_rank": len(positive),
        "cutoff": cutoff,
        "minimum_raw_eigenvalue": float(eigenvalues.min()),
        "maximum_raw_eigenvalue": float(eigenvalues.max()),
        "negative_roundoff_eigenvalue_count": int((eigenvalues < 0).sum()),
        "discarded_eigenvalue_count": int((~retained).sum()),
        "maximum_absolute_discarded_eigenvalue": float(np.max(np.abs(eigenvalues[~retained])))
        if (~retained).any()
        else 0.0,
        "symmetry_roundoff_error": symmetry_error,
        "maximum_spectral_reconstruction_absolute_difference": float(
            np.max(np.abs(reconstruction - gram))
        ),
        "maximum_training_query_factor_absolute_difference": float(
            np.max(np.abs(gram @ projection - factor))
        ),
        "labels_used_for_eigenbasis": False,
        "numerical_only_policy": (
            "Original PSD1e-9 relative gate; symmetry1e-12 roundoff average; "
            "discard eigenvalues<=1e-12*max(1,lambda_max). "
            "No output or label clipping/repair."
        ),
    }
    return factor, positive, vectors, projection, diagnostics


def ridge_initialization(y, eigenvalues, eigenvectors, alpha):
    target = original_targets(y, len(eigenvectors))
    mean = target.mean(0)
    theta = (
        np.sqrt(eigenvalues)[:, None]
        * (eigenvectors.T @ (target - mean))
        / (eigenvalues[:, None] + alpha)
    )
    initial = np.r_[theta.ravel(), mean]
    require(np.isfinite(initial).all(), "Same-partition direct Ridge initialization overflow")
    return initial


def _configuration(gamma, alpha):
    require(
        type(gamma) in (int, float)
        and gamma in (0.1, 1.0)
        and type(alpha) in (int, float)
        and alpha in (10.0, 1.0, 0.1),
        "Require explicit frozen gamma (.1/1) and alpha (10/1/.1)",
    )
    return {"gamma": float(gamma), "alpha": float(alpha)}


def _runtime():
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}


def _provenance(value):
    require(isinstance(value, dict), "Caller training provenance required")
    required = {
        "label_kind",
        "training_allowed",
        "source",
        "permission_evidence",
        "approved_by",
        "audit_basis",
        "use_scope",
        "public_redistribution",
    }
    require(
        required <= value.keys()
        and value["training_allowed"] is True
        and isinstance(value["label_kind"], str)
        and value["label_kind"] in ("synthetic", "teacher", "experimental", "published_reference")
        and value["audit_basis"] in ("operator_assumption", "documented_permission")
        and value["use_scope"] == "scientific_private_noncommercial"
        and value["public_redistribution"] is False
        and all(
            isinstance(value[k], str) and value[k].strip()
            for k in ("source", "permission_evidence", "approved_by")
        ),
        "Unapproved or mixed-label training provenance",
    )
    for flag in (
        "qualified_HSP_backend",
        "calibrated_HSP_uncertainty",
        "commercial_allowed",
        "raw_data_redistribution_allowed",
        "trained_weights_redistribution_allowed",
    ):
        require(value.get(flag, False) is False, "Caller provenance cannot broaden research scope")
    require(value.get("evaluation_only", False) is False, "Evaluation-only labels cannot train")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Require finite JSON caller provenance") from error
    return copy.deepcopy(value)


def _digest(state):
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _fit_state(features, target, config, provenance):
    preparation = BlockKernelPreparation.fit(features)
    rows = preparation.parameters["training_rows"]
    y = original_targets(target, rows)
    gram = _gram(preparation.coordinates, preparation.parameters, "mixed_physics", config["gamma"])
    factor, values, vectors, projection, spectral = spectral_basis(gram)
    alpha = config["alpha"]
    initial = ridge_initialization(target, values, vectors, alpha)
    initial_loss, _ = objective_gradient(initial, factor, y, alpha)
    scale = parameter_scales(values, alpha, rows)
    mapped = initial * scale
    latest = {}

    def recorded_objective(parameter):
        loss, original_gradient = objective_gradient(
            np.asarray(parameter, float) / scale, factor, y, alpha
        )
        optimizer_gradient = original_gradient / scale
        latest.update(
            last_finite_objective=loss,
            last_finite_original_gradient=original_gradient.tolist(),
            last_finite_optimizer_gradient=optimizer_gradient.tolist(),
            last_finite_original_gradient_infinity_norm=float(np.max(np.abs(original_gradient))),
            last_finite_optimizer_gradient_infinity_norm=float(np.max(np.abs(optimizer_gradient))),
        )
        return loss, optimizer_gradient

    try:
        result = minimize(
            recorded_objective,
            mapped,
            method="L-BFGS-B",
            jac=True,
            bounds=None,
            options=copy.deepcopy(OPTIMIZER_OPTIONS),
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise NumericalRefusal(
            "Optimizer raised before returning result; no retry",
            {
                "result_returned": False,
                "status": None,
                "message": str(error),
                "result_fun": None,
                "original_gradient_infinity_norm": None,
                "optimizer_gradient_infinity_norm": None,
                "objective_consistent": False,
                "optimizer_jacobian_consistent": False,
                "options": copy.deepcopy(OPTIMIZER_OPTIONS),
                "initial_objective": initial_loss,
                "numerical_training_version": 2,
                **latest,
            },
        ) from error
    certificate, original = optimizer_diagnostics(result, factor, y, alpha, values, initial_loss)
    certificate["initial_direct_Ridge_roundtrip_maximum_difference"] = float(
        np.max(np.abs(mapped / scale - initial))
    )
    if not certificate["success"] or certificate["status"] != 0:
        raise NumericalRefusal("L-BFGS-B did not converge; no retry/fallback", certificate)
    if (
        not certificate["result_coordinates_finite_and_aligned"]
        or not certificate["original_objective_gradient_finite"]
    ):
        raise NumericalRefusal("Nonfinite/unaligned returned optimizer state", certificate)
    if (
        not certificate["original_gradient_gate_pass"]
        or not certificate["objective_consistent"]
        or not certificate["optimizer_jacobian_consistent"]
    ):
        raise NumericalRefusal(
            "Original-coordinate gradient/objective consistency gate failed; no retry", certificate
        )
    if not certificate["objective_not_increased"]:
        raise NumericalRefusal(
            "Final original objective exceeds fixed Ridge initialization", certificate
        )
    rank = len(values)
    state = {
        "schema_version": 1,
        "config": config,
        "preparation": copy.deepcopy(preparation.parameters),
        "eigenvalues_retained": values.tolist(),
        "eigenvectors_retained": vectors.tolist(),
        "query_projection": projection.tolist(),
        "theta_signed": original[: 3 * rank].reshape(rank, 3).tolist(),
        "intercept_signed_unpenalized": original[3 * rank :].tolist(),
        "spectral_diagnostics": spectral,
        "numerical_training_version": 2,
        "optimizer_diagnostics_before_admissibility": certificate,
        "delta_unit_MPa_half": DELTA_UNIT_MPA_HALF,
        "response": "learned_forward_softplus_signed_latent_v1",
        "target_formulation": "original_delta_in_loss",
        "zero_training_targets_preserved": True,
        "array_level_only": True,
        "qualified_HSP_backend": False,
        "calibrated_HSP_uncertainty": False,
        "runtime_versions": _runtime(),
        "provenance": provenance,
        "compatibility_PCA_computed_and_stored_train_only_but_not_used_by_kernel": True,
        "state_digest_is_integrity_only_not_permission_or_accuracy_evidence": True,
    }
    state["state_sha256"] = _digest(state)
    return state


def _validate_state(state):
    require(
        isinstance(state, dict)
        and type(state.get("schema_version")) is int
        and state.get("schema_version") == 1
        and type(state.get("numerical_training_version")) is int
        and state.get("numerical_training_version") == 2
        and state.get("array_level_only") is True
        and state.get("qualified_HSP_backend") is False
        and state.get("calibrated_HSP_uncertainty") is False
        and state.get("response") == "learned_forward_softplus_signed_latent_v1"
        and state.get("target_formulation") == "original_delta_in_loss"
        and state.get("delta_unit_MPa_half") == 1.0
        and state.get("zero_training_targets_preserved") is True
        and state.get("compatibility_PCA_computed_and_stored_train_only_but_not_used_by_kernel")
        is True
        and state.get("state_digest_is_integrity_only_not_permission_or_accuracy_evidence") is True,
        "Unreviewed learned-forward research state",
    )
    require(state["state_sha256"] == _digest(state), "Research state integrity mismatch")
    require(state["runtime_versions"] == _runtime(), "Saved numerical runtime differs")
    _provenance(state["provenance"])
    c = state["config"]
    require(
        set(c) == {"gamma", "alpha"} and c == _configuration(c["gamma"], c["alpha"]),
        "Invalid saved configuration",
    )
    p = state["preparation"]
    rows = p["training_rows"]
    require(
        type(rows) is int
        and rows >= 3
        and p["fit_scope"] == "supplied_training_rows_only"
        and type(p["physics_columns"]) is int
        and p["physics_columns"] >= 1
        and set(p["blocks"]) == {"chemical", "embedding", "physics", "embedding_pca"},
        "Invalid training preparation scope",
    )
    # Existing transform validates imputer/scaler/PCA dimensions; PCA remains unused.
    _transform(
        {
            "chemical": np.zeros((1, 105)),
            "embedding": np.zeros((1, 768)),
            "physics": np.full((1, p["physics_columns"]), np.nan),
        },
        p,
    )
    for name, block in p["blocks"].items():
        training = _matrix(block["training"])
        trace = float(np.mean(np.sum(training**2, axis=1)))
        dimensions = len(block["scaling"]["mean"])
        require(
            training.shape == (rows, dimensions)
            and type(block["active"]) is bool
            and block["active"] == (trace > 0)
            and np.isfinite(block["coordinate_trace"])
            and np.isclose(trace, block["coordinate_trace"], rtol=1e-12, atol=0),
            "Invalid saved training trace/dimensions",
        )
        if name in ("chemical", "embedding"):
            require(
                dimensions == {"chemical": 105, "embedding": 768}[name],
                "Saved full feature dimensions differ",
            )
    values, vectors, projection, theta, intercept = (
        np.asarray(state[k], float)
        for k in (
            "eigenvalues_retained",
            "eigenvectors_retained",
            "query_projection",
            "theta_signed",
            "intercept_signed_unpenalized",
        )
    )
    require(values.ndim == 1 and 0 < len(values) <= rows, "Invalid retained rank")
    rank, spectral = len(values), state["spectral_diagnostics"]
    require(
        spectral["retained_rank"] == rank
        and spectral["training_rows"] == rows
        and spectral["labels_used_for_eigenbasis"] is False
        and np.isfinite(spectral["cutoff"])
        and spectral["cutoff"] > 0
        and np.all(values > spectral["cutoff"])
        and vectors.shape == projection.shape == (rows, rank)
        and theta.shape == (rank, 3)
        and intercept.shape == (3,)
        and all(np.isfinite(x).all() for x in (values, vectors, projection, theta, intercept))
        and np.allclose(vectors.T @ vectors, np.eye(rank), atol=1e-8, rtol=1e-8)
        and np.allclose(projection, vectors / np.sqrt(values), atol=1e-12, rtol=1e-12),
        "Invalid saved signed spectral state",
    )
    # Reconstruct only the label-independent saved training kernel: no eigenfit.
    coordinates = {
        name: np.asarray(p["blocks"][name]["training"], float)
        for name in ("chemical", "embedding", "physics")
    }
    kernel = _gram(coordinates, p, "mixed_physics", c["gamma"])
    amplitude = max(1.0, float(np.max(np.abs(kernel))))
    symmetry_error = float(np.max(np.abs(kernel - kernel.T)))
    eigen_scale = max(1.0, float(values[-1]))
    factor = vectors * np.sqrt(values)
    reconstruction_error = float(np.max(np.abs(factor @ factor.T - kernel)))
    query_error = float(np.max(np.abs(kernel @ projection - factor)))
    require(
        np.all(np.diff(values) >= 0)
        and symmetry_error <= SYMMETRY_RELATIVE_TOLERANCE * amplitude
        and np.isclose(spectral["cutoff"], EIGEN_CUTOFF_RELATIVE * eigen_scale, rtol=1e-12, atol=0)
        and np.isclose(spectral["maximum_raw_eigenvalue"], values[-1], rtol=1e-12, atol=0)
        and np.isfinite(spectral["minimum_raw_eigenvalue"])
        and spectral["minimum_raw_eigenvalue"] >= -PSD_RELATIVE_TOLERANCE * eigen_scale
        and spectral["discarded_eigenvalue_count"] == rows - rank
        and np.allclose(
            kernel @ vectors, vectors * values, rtol=1e-9, atol=PSD_RELATIVE_TOLERANCE * eigen_scale
        )
        and reconstruction_error <= rows * PSD_RELATIVE_TOLERANCE * eigen_scale
        and np.isclose(
            reconstruction_error,
            spectral["maximum_spectral_reconstruction_absolute_difference"],
            rtol=1e-5,
            atol=1e-12,
        )
        and np.isclose(
            query_error,
            spectral["maximum_training_query_factor_absolute_difference"],
            rtol=1e-5,
            atol=1e-12,
        ),
        "Saved spectral basis does not represent the saved training kernel",
    )
    diag = state["optimizer_diagnostics_before_admissibility"]
    require(
        diag["numerical_training_version"] == 2
        and diag["method"] == "L-BFGS-B"
        and diag["options"] == OPTIMIZER_OPTIONS
        and diag["success"] is True
        and type(diag["status"]) is int
        and diag["status"] == 0
        and diag["gradient_gate_original_coordinates"] == FINAL_GRADIENT_INFINITY_GATE
        and all(
            diag[k] is True
            for k in (
                "result_returned",
                "result_coordinates_finite_and_aligned",
                "original_objective_gradient_finite",
                "objective_consistent",
                "optimizer_jacobian_consistent",
                "original_gradient_gate_pass",
                "objective_not_increased",
                "local_convergence_only_no_global_minimum_claim",
            )
        ),
        "Invalid saved optimizer admissibility",
    )
    raw, scaled, reported = (
        np.asarray(diag[k], float)
        for k in ("original_gradient", "optimizer_gradient", "reported_optimizer_jacobian")
    )
    require(
        raw.shape == scaled.shape == reported.shape == (3 * (rank + 1),)
        and all(np.isfinite(x).all() for x in (raw, scaled, reported))
        and np.max(np.abs(raw)) <= FINAL_GRADIENT_INFINITY_GATE
        and all(
            np.max(np.abs(x)) == diag[k]
            for x, k in (
                (raw, "original_gradient_infinity_norm"),
                (scaled, "optimizer_gradient_infinity_norm"),
                (reported, "reported_optimizer_jacobian_infinity_norm"),
            )
        )
        and np.allclose(
            scaled, raw / parameter_scales(values, c["alpha"], rows), rtol=1e-12, atol=1e-12
        )
        and np.allclose(scaled, reported, rtol=1e-12, atol=1e-12),
        "Invalid saved original/optimizer gradient certificate",
    )
    losses = np.asarray(
        [diag[k] for k in ("initial_objective", "final_objective", "result_fun")], float
    )
    require(
        np.isfinite(losses).all()
        and np.all(losses >= 0)
        and np.isclose(losses[1], losses[2], rtol=1e-12, atol=1e-12)
        and losses[1] <= losses[0] + 1e-10 * max(1.0, losses[0]),
        "Invalid saved objective certificate",
    )


@dataclass
class SoftplusKernelHead:
    """Single explicit configuration for approved arrays; no HSP applicability claim.

    ``predict`` returns values, validity masks and refusal reasons. Targets and
    external asset files are absent from saved state. A digest checks integrity,
    not authenticity; rederiving a training certificate requires original targets.
    """

    parameters: dict

    @classmethod
    def fit(cls, features, target, *, gamma, alpha, provenance):
        reviewed = _provenance(provenance)
        config = _configuration(gamma, alpha)
        return cls.restore(_fit_state(features, target, config, reviewed))

    def state(self):
        return copy.deepcopy(self.parameters)

    @classmethod
    def restore(cls, state):
        try:
            state = copy.deepcopy(state)
            _validate_state(state)
        except (KeyError, TypeError, OverflowError, IndexError) as error:
            raise ValueError("Malformed learned-forward state") from error
        return cls(state)

    def predict(self, features):
        # Guard mutable in-memory state as well as explicit JSON restoration.
        try:
            _validate_state(self.parameters)
        except (KeyError, TypeError, OverflowError, IndexError) as error:
            raise ValueError("Malformed learned-forward state") from error
        state = self.parameters
        query = _transform(features, state["preparation"])
        kernel = _gram(query, state["preparation"], "mixed_physics", state["config"]["gamma"])
        with np.errstate(over="ignore", invalid="ignore"):
            latent = (kernel @ np.asarray(state["query_projection"])) @ np.asarray(
                state["theta_signed"]
            )
            latent += np.asarray(state["intercept_signed_unpenalized"])
        return response(latent)
