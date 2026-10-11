"""Small array-only research regressors; no qualified HSP inference or downloads.

Callers must enforce rights, chemical scope and family isolation before fitting.
These JSON states never authorize a physical backend or calibrated uncertainty.
"""

import copy
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

GRIDS = {
    "concat_ridge": tuple({"alpha": a} for a in (10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01)),
    "rbf_kernel": tuple(
        {"alpha": a, "gamma": g} for a in (10.0, 1.0, 0.1) for g in (0.1, 1.0, 10.0)
    ),
    "extra_trees": tuple({"max_depth": d, "min_samples_leaf": m} for d in (2, 4) for m in (8, 4)),
    "bottleneck_mlp": tuple({"width": w, "alpha": a} for w in (8, 16) for a in (100.0, 10.0)),
}


def _inputs(chemical, embedding, target=None):
    try:
        chemical, embedding = (np.asarray(v, dtype=float) for v in (chemical, embedding))
    except (TypeError, ValueError):
        raise ValueError("Require finite aligned feature matrices") from None
    if (
        chemical.ndim != 2
        or embedding.ndim != 2
        or len(chemical) != len(embedding)
        or not len(chemical)
        or not chemical.shape[1]
        or not embedding.shape[1]
        or not np.isfinite(chemical).all()
        or not np.isfinite(embedding).all()
    ):
        raise ValueError("Require finite aligned feature matrices")
    if target is not None:
        target = np.asarray(target, dtype=float)
        if (
            target.ndim != 2
            or target.shape != (len(chemical), 3)
            or len(chemical) < 3
            or not np.isfinite(target).all()
        ):
            raise ValueError("Require at least three rows with finite three-component targets")
    return chemical, embedding, target


def _scale_state(values):
    scaler = StandardScaler().fit(values)
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def _scaled(values, state):
    mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
    if mean.shape != (values.shape[1],) or scale.shape != mean.shape or not (scale > 0).all():
        raise ValueError("Preprocessing dimensions or scale differ")
    return (values - mean) / scale


def fit_fusion(chemical, embedding):
    """Fit chemical scaling and at most eight embedding PCs on supplied rows only."""
    chemical, embedding, _ = _inputs(chemical, embedding)
    if len(chemical) < 3:
        raise ValueError("Fusion needs at least three training rows")
    cs, es = _scale_state(chemical), _scale_state(embedding)
    e = _scaled(embedding, es)
    pca = PCA(n_components=min(8, len(e) - 1, e.shape[1]), svd_solver="full").fit(e)
    components, mean = pca.components_, pca.mean_
    scores = (e - mean) @ components.T
    state = {
        "chemical": cs,
        "embedding": es,
        "pca_mean": mean.tolist(),
        "pca_components": components.tolist(),
        "scores": _scale_state(scores),
    }
    return np.c_[_scaled(chemical, cs), _scaled(scores, state["scores"])], state


def transform_fusion(chemical, embedding, state):
    chemical, embedding, _ = _inputs(chemical, embedding)
    e = _scaled(embedding, state["embedding"])
    components, mean = np.asarray(state["pca_components"]), np.asarray(state["pca_mean"])
    if components.ndim != 2 or components.shape[1] != e.shape[1] or mean.shape != (e.shape[1],):
        raise ValueError("Embedding PCA dimensions differ")
    scores = (e - mean) @ components.T
    output = np.c_[_scaled(chemical, state["chemical"]), _scaled(scores, state["scores"])]
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite fused features")
    return output


def _kernel(query, training, gamma):
    distance = np.maximum(
        np.sum(query**2, axis=1)[:, None]
        + np.sum(training**2, axis=1)[None, :]
        - 2 * query @ training.T,
        0.0,
    )
    return np.exp(-gamma * distance)


def _tree_prediction(x, state):
    left, right, feature = (np.asarray(state[k], dtype=int) for k in ("left", "right", "feature"))
    threshold, value = np.asarray(state["threshold"]), np.asarray(state["value"])
    count = len(left)
    if right.shape != left.shape or feature.shape != left.shape or value.shape != (count, 3):
        raise ValueError("Tree dimensions differ")
    current = np.zeros(len(x), dtype=int)
    for _ in range(count + 1):
        if (current < 0).any() or (current >= count).any():
            raise ValueError("Tree child index outside state")
        terminal = (left[current] == -1) & (right[current] == -1)
        if terminal.all():
            return value[current]
        rows = np.flatnonzero(~terminal)
        node, columns = current[rows], feature[current[rows]]
        if (columns < 0).any() or (columns >= x.shape[1]).any():
            raise ValueError("Tree feature index outside input")
        current[rows] = np.where(x[rows, columns] <= threshold[node], left[node], right[node])
    raise ValueError("Tree contains a cycle")


@dataclass
class ResearchHead:
    """A parameter state for already qualified research arrays, not an HSPModel."""

    parameters: dict

    @classmethod
    def fit(cls, chemical, embedding, target, kind, config):
        chemical, embedding, target = _inputs(chemical, embedding, target)
        if kind not in GRIDS or config not in GRIDS[kind]:
            raise ValueError("Architecture/configuration is outside the small reviewed grid")
        x, fusion = fit_fusion(chemical, embedding)
        ys = _scale_state(target)
        y = _scaled(target, ys)
        optimization = {"convergence_warnings": []}
        if kind == "concat_ridge":
            fitted = Ridge(alpha=config["alpha"]).fit(x, y)
            backend = {"coef": fitted.coef_.tolist(), "intercept": fitted.intercept_.tolist()}
        elif kind == "rbf_kernel":
            gamma = config["gamma"] / x.shape[1]
            fitted = KernelRidge(alpha=config["alpha"], kernel="rbf", gamma=gamma).fit(x, y)
            backend = {
                "training_features": x.tolist(),
                "dual_coef": fitted.dual_coef_.tolist(),
                "gamma": gamma,
            }
        elif kind == "extra_trees":
            fitted = ExtraTreesRegressor(n_estimators=128, **config, random_state=42, n_jobs=1).fit(
                x, y
            )
            trees = []
            for estimator in fitted.estimators_:
                tree = estimator.tree_
                trees.append(
                    {
                        "left": tree.children_left.tolist(),
                        "right": tree.children_right.tolist(),
                        "feature": tree.feature.tolist(),
                        "threshold": tree.threshold.tolist(),
                        "value": tree.value.reshape(tree.node_count, 3).tolist(),
                    }
                )
            backend = {"trees": trees}
        else:
            fitted = MLPRegressor(
                hidden_layer_sizes=(config["width"],),
                activation="tanh",
                solver="lbfgs",
                alpha=config["alpha"],
                max_iter=600,
                tol=1e-5,
                random_state=42,
                early_stopping=False,
            )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                fitted.fit(x, y)
            optimization = {
                "convergence_warnings": [str(w.message) for w in recorded],
                "iterations": int(fitted.n_iter_),
                "loss": float(fitted.loss_),
                "fixed_iteration_budget": 600,
            }
            backend = {
                "coefs": [a.tolist() for a in fitted.coefs_],
                "intercepts": [a.tolist() for a in fitted.intercepts_],
            }
        state = {
            "schema_version": 1,
            "kind": kind,
            "config": copy.deepcopy(config),
            "fusion": fusion,
            "target_scaling": ys,
            "backend": backend,
            "optimization": optimization,
            "training_rows": len(x),
            "qualified_hsp_backend": False,
            "calibrated_uncertainty": False,
        }
        head = cls.restore(state)
        expected = fitted.predict(x) * np.asarray(ys["scale"]) + np.asarray(ys["mean"])
        if not np.allclose(head.predict(chemical, embedding), expected, atol=1e-9, rtol=1e-12):
            raise ValueError("Native estimator and JSON research head disagree")
        return head

    def state(self):
        return copy.deepcopy(self.parameters)

    @classmethod
    def restore(cls, state):
        if (
            state.get("schema_version") != 1
            or state.get("kind") not in GRIDS
            or state.get("config") not in GRIDS[state["kind"]]
            or state.get("qualified_hsp_backend") is not False
            or state.get("calibrated_uncertainty") is not False
        ):
            raise ValueError("Unreviewed research head state")
        return cls(copy.deepcopy(state))

    def predict(self, chemical, embedding):
        state = self.parameters
        x = transform_fusion(chemical, embedding, state["fusion"])
        backend, kind = state["backend"], state["kind"]
        if kind == "concat_ridge":
            output = x @ np.asarray(backend["coef"]).T + np.asarray(backend["intercept"])
        elif kind == "rbf_kernel":
            output = _kernel(x, np.asarray(backend["training_features"]), backend["gamma"]) @ (
                np.asarray(backend["dual_coef"])
            )
        elif kind == "extra_trees":
            # sklearn trees compare float32 inputs; preserve threshold decisions on reload.
            output = np.mean(
                [_tree_prediction(x.astype(np.float32), t) for t in backend["trees"]], 0
            )
        else:
            output = x
            for i, (coef, intercept) in enumerate(
                zip(backend["coefs"], backend["intercepts"], strict=True)
            ):
                output = output @ np.asarray(coef) + np.asarray(intercept)
                if i < len(backend["coefs"]) - 1:
                    output = np.tanh(output)
        output = output * np.asarray(state["target_scaling"]["scale"]) + np.asarray(
            state["target_scaling"]["mean"]
        )
        if output.shape != (len(x), 3) or not np.isfinite(output).all():
            raise ValueError("Nonfinite or misaligned research predictions")
        return output
