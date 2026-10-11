"""Finite additive kernels for reviewed research arrays; no physical qualification."""

import copy
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .theory_heads import MedianFeatures

TEMPLATES = {
    "chemical_linear": (("chemical", "linear", 1.0),),
    "linear_fusion": (("chemical", "linear", 0.5), ("embedding", "linear", 0.5)),
    "rbf_pca": (("chemical", "rbf", 0.5), ("embedding_pca", "rbf", 0.5)),
    "mixed_physics": (
        ("chemical", "linear", 0.4),
        ("embedding", "rbf", 0.4),
        ("physics", "rbf", 0.2),
    ),
}
KERNEL_GRID = tuple(
    {"template": template, "alpha": alpha, "gamma": gamma}
    for template in TEMPLATES
    for gamma in ((None,) if template in ("chemical_linear", "linear_fusion") else (0.1, 1.0))
    for alpha in (10.0, 1.0, 0.1)
)
NUMERIC_CAP = 1e100


def _matrix(values, *, missing=False):
    values = np.asarray(values, float)
    if (
        values.ndim != 2
        or not len(values)
        or np.isinf(values).any()
        or (not missing and np.isnan(values).any())
        or (np.abs(values[np.isfinite(values)]) > NUMERIC_CAP).any()
    ):
        raise ValueError("Require aligned matrices within the finite numerical cap")
    return values


def _inputs(features, *, physics_columns=None):
    if not isinstance(features, dict) or set(features) != {"chemical", "embedding", "physics"}:
        raise ValueError("Require chemical, embedding and physics feature blocks")
    blocks = {key: _matrix(value, missing=key == "physics") for key, value in features.items()}
    c, e, p = (blocks[key] for key in ("chemical", "embedding", "physics"))
    if (
        c.shape[1] != 105
        or e.shape[1] != 768
        or p.shape[1] < 1
        or len(c) != len(e)
        or len(c) != len(p)
        or (physics_columns is not None and p.shape[1] != physics_columns)
    ):
        raise ValueError("Require aligned 105 chemical / 768 embedding columns and physics columns")
    return blocks


def _scale_fit(values):
    if not values.shape[1]:
        return values.copy(), {"mean": [], "scale": []}
    scaler = StandardScaler().fit(values)
    constant = np.all(values == values[0], axis=0)
    scaler.mean_[constant], scaler.scale_[constant] = values[0, constant], 1.0
    state = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    return _scale(values, state), state


def _scale(values, state):
    mean, scale = (np.asarray(state[key], float) for key in ("mean", "scale"))
    if (
        mean.shape != (values.shape[1],)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("Invalid training scaling state")
    return _matrix((values - mean) / scale)


def _transform(features, preparation):
    blocks = _inputs(features, physics_columns=preparation["physics_columns"])
    imputer = preparation["physics_imputer"]
    p = (
        MedianFeatures.restore(imputer).transform(blocks["physics"])
        if imputer is not None
        else np.empty((len(blocks["physics"]), 0))
    )
    output = {
        key: _scale(p if key == "physics" else blocks[key], state["scaling"])
        for key, state in preparation["blocks"].items()
        if key != "embedding_pca"
    }
    pca = preparation["pca"]
    components, mean = np.asarray(pca["components"], float), np.asarray(pca["mean"], float)
    if (
        components.ndim != 2
        or components.shape[1] != 768
        or not 1 <= len(components) <= 8
        or mean.shape != (768,)
        or not np.isfinite(components).all()
        or not np.isfinite(mean).all()
    ):
        raise ValueError("Invalid training PCA state")
    scores = (output["embedding"] - mean) @ components.T
    output["embedding_pca"] = _scale(scores, preparation["blocks"]["embedding_pca"]["scaling"])
    return output


def _kernel(query, training, block, kind, gamma):
    if not block["active"]:
        return np.zeros((len(query), len(training)))
    if kind == "linear":
        return query @ training.T / block["coordinate_trace"]
    squared = np.maximum(
        np.sum(query**2, 1)[:, None] + np.sum(training**2, 1)[None, :] - 2 * query @ training.T,
        0.0,
    )
    return np.exp(-gamma * squared / training.shape[1])


def _gram(query, preparation, template, gamma):
    blocks = preparation["blocks"]
    if not any(blocks[name]["active"] for name, _, _ in TEMPLATES[template]):
        raise ValueError("All blocks in this template have zero training trace")
    terms = [
        weight
        * _kernel(query[name], np.asarray(blocks[name]["training"]), blocks[name], kind, gamma)
        for name, kind, weight in TEMPLATES[template]
    ]
    output = np.sum(terms, axis=0)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite additive kernel")
    return output


@dataclass
class BlockKernelPreparation:
    """Fit once per training partition; reuse eigendecompositions across alphas/targets."""

    parameters: dict
    coordinates: dict = field(repr=False)
    decompositions: dict = field(default_factory=dict, repr=False)

    @classmethod
    def fit(cls, features):
        blocks = _inputs(features)
        if len(blocks["chemical"]) < 3:
            raise ValueError("At least three training rows are required")
        p = blocks["physics"]
        imputer = MedianFeatures.fit(p) if np.isfinite(p).any() else None
        imputed = imputer.transform(p) if imputer is not None else np.empty((len(p), 0))
        coordinates, scales = {}, {}
        for key in ("chemical", "embedding", "physics"):
            coordinates[key], scales[key] = _scale_fit(imputed if key == "physics" else blocks[key])
        e = coordinates["embedding"]
        count = min(8, len(e) - 1)
        if np.any(e):
            pca = PCA(n_components=count, svd_solver="full").fit(e)
            components, mean = pca.components_, pca.mean_
        else:
            components, mean = np.zeros((count, 768)), np.zeros(768)
        coordinates["embedding_pca"], scales["embedding_pca"] = _scale_fit(
            (e - mean) @ components.T
        )
        prepared = {}
        for key, values in coordinates.items():
            trace = float(np.mean(np.sum(values**2, axis=1)))
            prepared[key] = {
                "scaling": scales[key],
                "training": values.tolist(),
                "coordinate_trace": trace,
                "active": trace > 0,
            }
        return cls(
            {
                "fit_scope": "supplied_training_rows_only",
                "training_rows": len(e),
                "physics_columns": p.shape[1],
                "physics_imputer": imputer.state() if imputer is not None else None,
                "pca": {"components": components.tolist(), "mean": mean.tolist()},
                "blocks": prepared,
            },
            coordinates,
        )

    def fit_head(self, target, config):
        if config not in KERNEL_GRID:
            raise ValueError("Unreviewed kernel configuration")
        target = _matrix(target)
        if target.shape != (self.parameters["training_rows"], 3):
            raise ValueError("Require aligned three-component training targets")
        key = (config["template"], config["gamma"])
        if key not in self.decompositions:
            gram = _gram(self.coordinates, self.parameters, *key)
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            if eigenvalues.min() < -1e-9 * max(1.0, eigenvalues.max()):
                raise ValueError("Additive kernel is not positive semidefinite")
            self.decompositions[key] = eigenvalues, eigenvectors
        values, vectors = self.decompositions[key]
        denominator = values + config["alpha"]
        if not (denominator > 0).all():
            raise ValueError("Nonpositive regularized kernel eigenvalue")
        mean = target.mean(0)
        dual = vectors @ ((vectors.T @ (target - mean)) / denominator[:, None])
        _matrix(dual)
        return BlockKernelHead(
            {
                "schema_version": 1,
                "config": copy.deepcopy(config),
                "preparation": self.parameters,
                "dual": dual.tolist(),
                "target_mean": mean.tolist(),
                "qualified_hsp_backend": False,
                "calibrated_uncertainty": False,
            }
        )


@dataclass
class BlockKernelHead:
    """JSON state for already reviewed arrays, without chemical or rights qualification."""

    parameters: dict

    @classmethod
    def fit(cls, features, target, config):
        return BlockKernelPreparation.fit(features).fit_head(target, config)

    def state(self):
        return copy.deepcopy(self.parameters)

    @classmethod
    def restore(cls, state):
        if (
            state.get("schema_version") != 1
            or state.get("config") not in KERNEL_GRID
            or state.get("qualified_hsp_backend") is not False
            or state.get("calibrated_uncertainty") is not False
        ):
            raise ValueError("Unreviewed block-kernel state")
        state = copy.deepcopy(state)
        p = state["preparation"]
        if p["fit_scope"] != "supplied_training_rows_only" or p["training_rows"] < 3:
            raise ValueError("Invalid training preparation scope")
        if set(p["blocks"]) != {"chemical", "embedding", "physics", "embedding_pca"}:
            raise ValueError("Invalid saved feature blocks")
        for block in p["blocks"].values():
            training = _matrix(block["training"])
            trace = float(np.mean(np.sum(training**2, axis=1)))
            if (
                len(training) != p["training_rows"]
                or type(block["active"]) is not bool
                or block["active"] != (trace > 0)
                or not np.isclose(trace, block["coordinate_trace"], rtol=1e-12, atol=0)
            ):
                raise ValueError("Invalid saved training trace")
        head = cls(state)
        head.predict(
            {
                "chemical": np.zeros((1, 105)),
                "embedding": np.zeros((1, 768)),
                "physics": np.full((1, p["physics_columns"]), np.nan),
            }
        )
        return head

    def predict(self, features):
        state = self.parameters
        query = _transform(features, state["preparation"])
        config = state["config"]
        gram = _gram(query, state["preparation"], config["template"], config["gamma"])
        dual, mean = _matrix(state["dual"]), np.asarray(state["target_mean"], float)
        if dual.shape != (gram.shape[1], 3) or mean.shape != (3,) or not np.isfinite(mean).all():
            raise ValueError("Invalid saved target dimensions")
        return _matrix(gram @ dual + mean)
