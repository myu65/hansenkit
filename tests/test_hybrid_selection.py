import numpy as np
import pytest

from hansenkit.models import LinearHead
from hansenkit.selection import _hybrid_fold_candidates, fit_group_selected_hybrid


def population():
    rng = np.random.default_rng(41)
    x, e = rng.normal(size=(30, 3)), rng.normal(size=(30, 4))
    y = np.column_stack([x[:, 0] + e[:, 0], np.full(30, 2.0), x[:, 1] - e[:, 2]])
    return x, e, y, np.repeat(np.arange(10), 3)


def test_nested_family_isolation_and_json_head_roundtrips():
    x, e, y, groups = population()
    base, residual, report = fit_group_selected_hybrid(x, e, y, groups, alphas=[0.1, 100.0])
    prediction = base.predict(x) + residual.predict(e)
    np.testing.assert_array_equal(
        prediction,
        LinearHead.restore(base.state()).predict(x)
        + LinearHead.restore(residual.state()).predict(e),
    )
    assert report["selected_candidates"][1] == "none"
    assert report["independent_accuracy_validated"] is False
    seen = np.zeros(len(x), dtype=int)
    for fold in report["outer_folds"]:
        train, held = fold["train_indices"], fold["validation_indices"]
        assert set(groups[train]).isdisjoint(groups[held])
        seen[held] += 1
        inner_seen = np.zeros(len(x), dtype=int)
        for inner in fold["inner_residual_folds"]:
            it, iv = inner["train_indices"], inner["validation_indices"]
            assert set(it + iv) <= set(train)
            assert set(groups[it]).isdisjoint(set(groups[iv]) | set(groups[held]))
            inner_seen[iv] += 1
        np.testing.assert_array_equal(inner_seen[train], 1)
        assert not inner_seen[held].any()
    np.testing.assert_array_equal(seen, 1)


def test_outer_held_labels_cannot_change_base_or_inner_residual_candidates():
    x, e, y, groups = population()
    train, held = np.arange(24), np.arange(24, 30)

    def run(targets):
        calls = []

        def base(ids):
            assert set(ids) <= set(train)
            calls.append(ids.copy())
            return LinearHead.fit(x[ids], targets[ids], 10.0), {}

        predictions, base_prediction, _ = _hybrid_fold_candidates(
            x, e, targets, groups, train, held, base, ("none", "constant", 10.0)
        )
        return predictions, base_prediction, calls

    first, initial_base, initial_calls = run(y)
    changed = y.copy()
    changed[held] += 1e6
    second, changed_base, changed_calls = run(changed)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(initial_base, changed_base)
    for one, two in zip(initial_calls, changed_calls, strict=True):
        np.testing.assert_array_equal(one, two)


@pytest.mark.parametrize("problem", ["too_few_families", "short_embedding", "nan_embedding"])
def test_invalid_hybrid_training_is_rejected(problem):
    x, e, y, groups = population()
    if problem == "too_few_families":
        groups = np.repeat(np.arange(3), 10)
    elif problem == "short_embedding":
        e = e[:-1]
    else:
        e[0, 0] = np.nan
    with pytest.raises(ValueError, match="families|finite aligned"):
        fit_group_selected_hybrid(x, e, y, groups, alphas=[1.0])
