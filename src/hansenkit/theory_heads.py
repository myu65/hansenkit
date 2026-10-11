"""Train-only missingness and positive component forests for array research.

The squared-target link is a mathematical prior, not computed cohesive energy.
Callers retain source rights, family isolation and provenance outside this API.
"""

import copy
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .research_heads import fit_fusion, transform_fusion

TREE_GRID = tuple({"max_depth": d, "min_samples_leaf": m} for d in (2, 4) for m in (8, 4))


def _features(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or not len(values) or np.isinf(values).any():
        raise ValueError("Require a nonempty feature matrix; NaN is the only missing marker")
    return values


@dataclass(frozen=True)
class MedianFeatures:
    """Drop wholly missing training columns; append masks to train-median imputation."""

    parameters: dict

    @classmethod
    def fit(cls, values):
        values = _features(values)
        active = np.flatnonzero(np.isfinite(values).any(0))
        if not len(active):
            raise ValueError("No observed training features")
        return cls(
            {
                "input_columns": values.shape[1],
                "active_columns": active.tolist(),
                "training_medians": np.nanmedian(values[:, active], axis=0).tolist(),
                "fit_scope": "supplied_training_rows_only",
            }
        )

    def transform(self, values):
        values = _features(values)
        state = self.parameters
        active = np.asarray(state["active_columns"], int)
        medians = np.asarray(state["training_medians"], float)
        if (
            values.shape[1] != state["input_columns"]
            or active.ndim != 1
            or medians.shape != active.shape
            or not len(active)
            or len(set(active)) != len(active)
            or (active < 0).any()
            or (active >= values.shape[1]).any()
            or not np.isfinite(medians).all()
            or state.get("fit_scope") != "supplied_training_rows_only"
        ):
            raise ValueError("Invalid train-only missing-feature state")
        selected = values[:, active]
        missing = np.isnan(selected)
        return np.c_[np.where(missing, medians, selected), missing.astype(float)]

    def state(self):
        return copy.deepcopy(self.parameters)

    @classmethod
    def restore(cls, state):
        instance = cls(copy.deepcopy(state))
        instance.transform(np.full((1, state["input_columns"]), np.nan))
        return instance


def _tree_predict(x, tree):
    left, right, features = (np.asarray(tree[k], int) for k in ("left", "right", "features"))
    thresholds, values = np.asarray(tree["thresholds"]), np.asarray(tree["values"])
    count = len(left)
    if (
        right.shape != (count,)
        or features.shape != (count,)
        or thresholds.shape != (count,)
        or values.shape != (count,)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Invalid positive tree state")
    nodes = np.zeros(len(x), int)
    for _ in range(count + 1):
        if (nodes < 0).any() or (nodes >= count).any():
            raise ValueError("Tree child outside state")
        terminal = (left[nodes] == -1) & (right[nodes] == -1)
        if terminal.all():
            return values[nodes]
        rows = np.flatnonzero(~terminal)
        current, columns = nodes[rows], features[nodes[rows]]
        if (columns < 0).any() or (columns >= x.shape[1]).any():
            raise ValueError("Tree feature outside input")
        nodes[rows] = np.where(
            x[rows, columns] <= thresholds[current], left[current], right[current]
        )
    raise ValueError("Tree contains a cycle")


@dataclass(frozen=True)
class PositiveForest:
    """Three independent positive leaf-mean heads, direct delta or delta-squared."""

    parameters: dict

    @classmethod
    def fit(cls, chemical, embedding, target, config, *, squared=False):
        target = np.asarray(target, float)
        if (
            config not in TREE_GRID
            or type(squared) is not bool
            or target.shape != (len(chemical), 3)
            or not np.isfinite(target).all()
            or (target < 0).any()
        ):
            raise ValueError("Require reviewed forest config and finite nonnegative three targets")
        x, fusion = fit_fusion(chemical, embedding)
        fitted_target = target**2 if squared else target
        if not np.isfinite(fitted_target).all():
            raise ValueError("Squared targets overflow")
        components, expected = [], []
        for j in range(3):
            fitted = ExtraTreesRegressor(n_estimators=128, **config, random_state=42, n_jobs=1).fit(
                x, fitted_target[:, j]
            )
            expected.append(fitted.predict(x))
            components.append(
                [
                    {
                        "left": e.tree_.children_left.tolist(),
                        "right": e.tree_.children_right.tolist(),
                        "features": e.tree_.feature.tolist(),
                        "thresholds": e.tree_.threshold.tolist(),
                        "values": e.tree_.value.reshape(-1).tolist(),
                    }
                    for e in fitted.estimators_
                ]
            )
        state = {
            "schema_version": 1,
            "kind": "positive_component_forest",
            "squared_target": squared,
            "config": copy.deepcopy(config),
            "fusion": fusion,
            "components": components,
            "qualified_hsp_backend": False,
            "calibrated_uncertainty": False,
            "computed_cohesive_energy": False,
            "negative_prediction_clipping": False,
        }
        head = cls.restore(state)
        expected = np.array(expected).T
        if squared:
            expected = np.sqrt(expected)
        if not np.allclose(head.predict(chemical, embedding), expected, atol=1e-10, rtol=1e-12):
            raise ValueError("Native positive forest and JSON prediction disagree")
        return head

    def state(self):
        return copy.deepcopy(self.parameters)

    @classmethod
    def restore(cls, state):
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "positive_component_forest"
            or state.get("config") not in TREE_GRID
            or type(state.get("squared_target")) is not bool
            or state.get("qualified_hsp_backend") is not False
            or state.get("calibrated_uncertainty") is not False
            or state.get("computed_cohesive_energy") is not False
            or state.get("negative_prediction_clipping") is not False
            or len(state.get("components", [])) != 3
            or any(len(c) != 128 for c in state["components"])
        ):
            raise ValueError("Invalid unqualified positive forest state")
        return cls(copy.deepcopy(state))

    def predict(self, chemical, embedding):
        state = self.parameters
        x = transform_fusion(chemical, embedding, state["fusion"]).astype(np.float32)
        output = np.column_stack(
            [
                np.mean([_tree_predict(x, tree) for tree in component], axis=0)
                for component in state["components"]
            ]
        )
        if not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("Invalid positive prediction; never clip or repair")
        return np.sqrt(output) if state["squared_target"] else output
