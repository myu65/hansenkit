import numpy as np
import pytest

from hansenkit.models import LinearHead
from hansenkit.selection import fit_group_selected_ridge


def fixture():
    x = np.arange(36, dtype=float).reshape(18, 2)
    x[:, 1] = x[:, 0] ** 2
    y = np.column_stack([x[:, 0] * 2, np.full(18, 7.0), x[:, 1] * 0.5])
    return x, y, np.repeat([f"family-{i}" for i in range(6)], 3)


def test_component_selection_constant_and_roundtrip():
    x, y, groups = fixture()
    head, report = fit_group_selected_ridge(x, y, groups, alphas=[0.01, 10000.0])
    assert report["selected_alphas"] == [0.01, None, 0.01]
    assert report["independent_accuracy_validated"] is False
    assert report["calibrated_intervals"] is False
    assert np.count_nonzero(head.coef[1]) == 0
    np.testing.assert_allclose(head.predict(x)[:, 1], 7.0)
    np.testing.assert_array_equal(head.predict(x), LinearHead.restore(head.state()).predict(x))


def test_native_integer_grouping_keys_preserve_the_selected_fit():
    x, y, groups = fixture()
    one, first = fit_group_selected_ridge(x, y, groups)
    two, second = fit_group_selected_ridge(x, y, np.repeat(np.arange(6), 3))
    assert first["selected_alphas"] == second["selected_alphas"]
    np.testing.assert_array_equal(one.predict(x), two.predict(x))


def test_preprocessing_and_candidate_labels_exclude_each_validation_family(monkeypatch):
    x, y, groups = fixture()
    original_fit = LinearHead.fit
    calls = []

    def observed_fit(inputs, targets, alpha=1.0):
        calls.append((inputs.copy(), targets.copy()))
        head = original_fit(inputs, targets, alpha)
        np.testing.assert_allclose(head.mean, inputs.mean(axis=0))
        return head

    monkeypatch.setattr(LinearHead, "fit", observed_fit)
    _, report = fit_group_selected_ridge(x, y, groups, alphas=[0.01, 10000.0], folds=3)
    for fold, position in zip(report["folds"], range(0, 6, 2), strict=True):
        train, validation = fold["train_indices"], fold["validation_indices"]
        assert set(groups[train]).isdisjoint(groups[validation])
        for inputs, targets in calls[position : position + 2]:
            np.testing.assert_array_equal(inputs, x[train])
            np.testing.assert_array_equal(targets, y[train])


def test_family_macro_scores_do_not_overweight_large_validation_family():
    x, y, groups = fixture()
    indices = np.r_[np.repeat(np.arange(3), 10), np.arange(3, 18)]
    x, y, groups = x[indices], y[indices], groups[indices]
    _, report = fit_group_selected_ridge(x, y, groups, folds=3)
    absolute = np.empty_like(y)
    for fold in report["folds"]:
        train, validation = fold["train_indices"], fold["validation_indices"]
        absolute[validation] = np.abs(y[validation] - y[train].mean(axis=0))
    expected = np.mean([absolute[groups == key].mean(axis=0) for key in set(groups)], axis=0)
    np.testing.assert_allclose(report["candidate_scores"][0], expected)
    assert not np.allclose(expected, absolute.mean(axis=0))


@pytest.mark.parametrize("penalties", [[], [0], [-1], [np.inf], [np.nan]])
def test_invalid_penalties_rejected(penalties):
    with pytest.raises(ValueError, match="positive penalties"):
        fit_group_selected_ridge(*fixture(), alphas=penalties)


@pytest.mark.parametrize("problem", ["two_families", "blank_family", "nan_features", "nan_targets"])
def test_invalid_training_population_rejected(problem):
    x, y, groups = fixture()
    if problem == "two_families":
        groups = np.repeat(["one", "two"], 9)
    elif problem == "blank_family":
        groups[0] = ""
    elif problem == "nan_features":
        x[0, 0] = np.nan
    else:
        y[0, 0] = np.nan
    with pytest.raises(ValueError, match="families|finite aligned"):
        fit_group_selected_ridge(x, y, groups)
