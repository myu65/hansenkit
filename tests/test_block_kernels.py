import copy
import json

import numpy as np
import pytest

from hansenkit.block_kernels import (
    KERNEL_GRID,
    TEMPLATES,
    BlockKernelHead,
    BlockKernelPreparation,
)


def population(rows=16):
    rng = np.random.default_rng(921)
    features = {
        "chemical": rng.normal(size=(rows, 105)),
        "embedding": rng.normal(size=(rows, 768)),
        "physics": rng.normal(size=(rows, 4)),
    }
    features["physics"][::3, 1] = np.nan
    features["physics"][:, 3] = np.nan
    target = np.column_stack(
        (features["chemical"][:, 0] + 17, features["embedding"][:, 0] + 5, np.ones(rows) * 8)
    )
    return features, target


def native_coordinates(preparation, features):
    blocks = {key: value.copy() for key, value in features.items()}
    imputer = preparation["physics_imputer"]
    if imputer is None:
        blocks["physics"] = np.empty((len(blocks["physics"]), 0))
    else:
        p = blocks["physics"][:, imputer["active_columns"]]
        missing = np.isnan(p)
        blocks["physics"] = np.c_[
            np.where(missing, np.asarray(imputer["training_medians"]), p), missing.astype(float)
        ]
    for key in ("chemical", "embedding", "physics"):
        scaling = preparation["blocks"][key]["scaling"]
        blocks[key] = (blocks[key] - scaling["mean"]) / scaling["scale"]
    pca = preparation["pca"]
    scores = (blocks["embedding"] - pca["mean"]) @ np.asarray(pca["components"]).T
    scaling = preparation["blocks"]["embedding_pca"]["scaling"]
    blocks["embedding_pca"] = (scores - scaling["mean"]) / scaling["scale"]
    return blocks


def native_gram(preparation, coordinates, config):
    count = preparation["training_rows"]
    result = np.zeros((len(coordinates["chemical"]), count))
    for key, kind, weight in TEMPLATES[config["template"]]:
        block = preparation["blocks"][key]
        if not block["active"]:
            continue
        train, query = np.asarray(block["training"]), coordinates[key]
        if kind == "linear":
            matrix = query @ train.T / np.mean(np.sum(train**2, axis=1))
        else:
            # Deliberately use direct pair distances rather than project's norm expansion.
            distance = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=2)
            matrix = np.exp(-config["gamma"] * distance / train.shape[1])
        result += weight * matrix
    return result


@pytest.mark.parametrize("template", tuple(TEMPLATES))
def test_closed_form_native_prediction_and_json_roundtrip(template):
    features, target = population()
    query, _ = population(7)
    config = next(c for c in KERNEL_GRID if c["template"] == template and c["alpha"] == 1.0)
    head = BlockKernelHead.fit(features, target, config)
    state = head.state()
    preparation = state["preparation"]
    training = native_coordinates(preparation, features)
    query_coordinates = native_coordinates(preparation, query)
    gram = native_gram(preparation, training, config)
    coefficients = np.linalg.solve(gram + np.eye(len(target)), target - target.mean(0))
    expected = target.mean(0) + native_gram(preparation, query_coordinates, config) @ coefficients
    np.testing.assert_allclose(head.predict(query), expected, atol=1e-11, rtol=1e-12)
    restored = BlockKernelHead.restore(json.loads(json.dumps(state, allow_nan=False)))
    np.testing.assert_array_equal(head.predict(query), restored.predict(query))
    np.testing.assert_allclose(head.predict(query)[:, 2], 8)
    assert restored.state()["qualified_hsp_backend"] is False
    assert restored.state()["calibrated_uncertainty"] is False


def test_held_extremes_do_not_change_training_stats_or_missing_column_selection():
    features, target = population()
    preparation = BlockKernelPreparation.fit(features)
    before = copy.deepcopy(preparation.parameters)
    head = preparation.fit_head(target, KERNEL_GRID[-1])
    query = {key: value.copy() for key, value in features.items()}
    query["chemical"] *= 1e6
    query["embedding"] -= 1e6
    query["physics"][:, 3] = 1e6
    assert np.isfinite(head.predict(query)).all()
    assert preparation.parameters == before
    assert before["physics_imputer"]["active_columns"] == [0, 1, 2]
    for key in ("chemical", "embedding"):
        np.testing.assert_allclose(before["blocks"][key]["scaling"]["mean"], features[key].mean(0))


def test_decomposition_reused_for_all_alphas_and_targets(monkeypatch):
    features, target = population()
    preparation = BlockKernelPreparation.fit(features)
    calls = []
    original = np.linalg.eigh

    def counted(values):
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    configs = [c for c in KERNEL_GRID if c["template"] == "rbf_pca" and c["gamma"] == 0.1]
    for config in configs:
        preparation.fit_head(target, config).predict(features)
    preparation.fit_head(target * 2, configs[0])
    assert calls == [len(target)]


def test_inactive_training_blocks_keep_fixed_weights_and_reject_empty_template():
    features, target = population()
    features["chemical"][:] = 1.23
    features["physics"][:] = np.nan
    preparation = BlockKernelPreparation.fit(features)
    assert not preparation.parameters["blocks"]["chemical"]["active"]
    assert not preparation.parameters["blocks"]["physics"]["active"]
    with pytest.raises(ValueError, match="zero training trace"):
        preparation.fit_head(target, KERNEL_GRID[0])
    config = next(c for c in KERNEL_GRID if c["template"] == "mixed_physics")
    head = preparation.fit_head(target, config)
    state = head.state()
    gram = native_gram(state["preparation"], preparation.coordinates, config)
    np.testing.assert_allclose(np.diag(gram), 0.4)
    restored = BlockKernelHead.restore(json.loads(json.dumps(state)))
    np.testing.assert_array_equal(head.predict(features), restored.predict(features))


@pytest.mark.parametrize(
    "key, value", [("qualified_hsp_backend", True), ("calibrated_uncertainty", True)]
)
def test_restoration_rejects_unearned_qualification(key, value):
    features, target = population()
    state = BlockKernelHead.fit(features, target, KERNEL_GRID[0]).state()
    state[key] = value
    with pytest.raises(ValueError, match="Unreviewed"):
        BlockKernelHead.restore(state)


def test_restoration_rejects_corrupt_traces_dimensions_and_configuration():
    features, target = population()
    head = BlockKernelHead.fit(features, target, KERNEL_GRID[0])
    state = head.state()
    state["preparation"]["blocks"]["chemical"]["coordinate_trace"] *= 2
    with pytest.raises(ValueError, match="trace"):
        BlockKernelHead.restore(state)
    state = head.state()
    state["dual"][0] = [1, 2]
    with pytest.raises(ValueError):
        BlockKernelHead.restore(state)
    state = head.state()
    state["config"]["alpha"] = 0
    with pytest.raises(ValueError):
        BlockKernelHead.restore(state)


@pytest.mark.parametrize("mutation", ("chemical_nan", "embedding_inf", "cap", "shape", "target"))
def test_bad_inputs_are_rejected(mutation):
    features, target = population()
    if mutation == "chemical_nan":
        features["chemical"][0, 0] = np.nan
    elif mutation == "embedding_inf":
        features["embedding"][0, 0] = np.inf
    elif mutation == "cap":
        features["physics"][0, 0] = 1e101
    elif mutation == "shape":
        features["embedding"] = features["embedding"][:, :-1]
    else:
        target = target[:-1]
    with pytest.raises(ValueError):
        BlockKernelHead.fit(features, target, KERNEL_GRID[0])


def test_grid_is_finite_and_linear_gamma_is_not_redundant():
    assert len(KERNEL_GRID) == 18
    assert len({tuple(config.items()) for config in KERNEL_GRID}) == 18
    assert all(config["gamma"] is None for config in KERNEL_GRID if "linear" in config["template"])
