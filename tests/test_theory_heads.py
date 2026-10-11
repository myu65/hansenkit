import copy
import json

import numpy as np
import pytest

from hansenkit.theory_heads import TREE_GRID, MedianFeatures, PositiveForest


def test_missing_features_fit_only_training_and_drop_unobserved():
    train = np.array([[1, np.nan, np.nan], [3, 2, np.nan], [5, 4, np.nan]])
    imputer = MedianFeatures.fit(train)
    state = copy.deepcopy(imputer.state())
    output = imputer.transform([[np.nan, 10000, 50000]])
    np.testing.assert_array_equal(output, [[3, 10000, 1, 0]])
    assert state["active_columns"] == [0, 1]
    assert imputer.state() == state
    restored = MedianFeatures.restore(json.loads(json.dumps(state)))
    np.testing.assert_array_equal(restored.transform(train), imputer.transform(train))


def test_missing_imputer_rejects_infinity_and_no_observations():
    with pytest.raises(ValueError):
        MedianFeatures.fit([[np.inf]])
    with pytest.raises(ValueError, match="observed"):
        MedianFeatures.fit([[np.nan]])


def population():
    rng = np.random.default_rng(813)
    chemical, embedding = rng.normal(size=(24, 5)), rng.normal(size=(24, 12))
    target = np.column_stack((chemical[:, 0] ** 2, np.full(24, 2.0), np.zeros(24)))
    return chemical, embedding, target


@pytest.mark.parametrize("squared", [False, True])
def test_positive_component_heads_native_export_roundtrip(squared):
    chemical, embedding, target = population()
    head = PositiveForest.fit(chemical, embedding, target, TREE_GRID[0], squared=squared)
    restored = PositiveForest.restore(json.loads(json.dumps(head.state())))
    output = restored.predict(chemical * 1000, embedding - 1000)
    assert np.isfinite(output).all() and (output >= 0).all()
    np.testing.assert_allclose(output[:, 1:], np.tile([2.0, 0.0], (24, 1)))
    np.testing.assert_array_equal(output, head.predict(chemical * 1000, embedding - 1000))
    assert restored.state()["negative_prediction_clipping"] is False
    assert restored.state()["computed_cohesive_energy"] is False


def test_negative_reference_targets_cannot_be_squared_into_valid_energy():
    chemical, embedding, target = population()
    target[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        PositiveForest.fit(chemical, embedding, target, TREE_GRID[0], squared=True)


def test_corrupted_positive_leaf_and_physical_declaration_rejected():
    chemical, embedding, target = population()
    head = PositiveForest.fit(chemical, embedding, target, TREE_GRID[0])
    state = head.state()
    state["qualified_hsp_backend"] = True
    with pytest.raises(ValueError):
        PositiveForest.restore(state)
    state = head.state()
    state["components"][0][0]["values"][0] = -1
    with pytest.raises(ValueError, match="positive"):
        PositiveForest.restore(state).predict(chemical, embedding)
