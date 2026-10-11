import copy
import json

import numpy as np
import pytest

from hansenkit.research_heads import GRIDS, ResearchHead, fit_fusion, transform_fusion


def population():
    rng = np.random.default_rng(82)
    chemical = rng.normal(size=(36, 4))
    embedding = rng.normal(size=(36, 12))
    target = np.column_stack(
        [chemical[:, 0] ** 2 + 10, chemical[:, 1] + embedding[:, 0], np.full(36, 4.0)]
    )
    return chemical, embedding, target


@pytest.mark.parametrize("kind", GRIDS)
def test_native_estimator_export_and_json_roundtrip(kind):
    chemical, embedding, target = population()
    head = ResearchHead.fit(chemical[:28], embedding[:28], target[:28], kind, GRIDS[kind][0])
    restored = ResearchHead.restore(json.loads(json.dumps(head.state())))
    np.testing.assert_array_equal(
        head.predict(chemical[28:], embedding[28:]),
        restored.predict(chemical[28:], embedding[28:]),
    )
    # The finite-budget MLP optimizes approximately; JSON restoration remains exact above.
    tolerance = 1e-4 if kind == "bottleneck_mlp" else 1e-8
    np.testing.assert_allclose(head.predict(chemical, embedding)[:, 2], 4.0, atol=tolerance)
    assert head.state()["qualified_hsp_backend"] is False
    assert head.state()["calibrated_uncertainty"] is False


def test_fusion_keeps_train_statistics_with_extreme_held_features():
    chemical, embedding, _ = population()
    train, state = fit_fusion(chemical[:28], embedding[:28])
    held_chemical, held_embedding = chemical[28:] + 1e5, embedding[28:] - 1e5
    original = copy.deepcopy(state)
    transform_fusion(held_chemical, held_embedding, state)
    assert state == original
    np.testing.assert_allclose(state["chemical"]["mean"], chemical[:28].mean(0))
    np.testing.assert_allclose(state["embedding"]["mean"], embedding[:28].mean(0))
    np.testing.assert_allclose(transform_fusion(chemical[:28], embedding[:28], state), train)
    assert len(state["pca_components"]) == 8


@pytest.mark.parametrize("problem", ["nan", "short", "wrong_targets", "unknown_config"])
def test_invalid_fit_inputs_and_unreviewed_config_fail(problem):
    chemical, embedding, target = population()
    config = GRIDS["concat_ridge"][0]
    if problem == "nan":
        chemical[0, 0] = np.nan
    elif problem == "short":
        embedding = embedding[:-1]
    elif problem == "wrong_targets":
        target = target[:, :2]
    else:
        config = {"alpha": -1}
    with pytest.raises(ValueError):
        ResearchHead.fit(chemical, embedding, target, "concat_ridge", config)


def test_json_state_cannot_declare_physical_qualification():
    chemical, embedding, target = population()
    head = ResearchHead.fit(chemical, embedding, target, "concat_ridge", GRIDS["concat_ridge"][0])
    state = head.state()
    state["qualified_hsp_backend"] = True
    with pytest.raises(ValueError, match="Unreviewed"):
        ResearchHead.restore(state)


def test_corrupt_tree_cycle_cannot_hang_prediction():
    chemical, embedding, target = population()
    head = ResearchHead.fit(chemical, embedding, target, "extra_trees", GRIDS["extra_trees"][0])
    state = head.state()
    tree = state["backend"]["trees"][0]
    tree["left"][0] = tree["right"][0] = 0
    with pytest.raises(ValueError, match="cycle"):
        ResearchHead.restore(state).predict(chemical, embedding)
