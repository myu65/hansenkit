"""Array-level selection on a caller's approved training partition only.

Callers must supply already reconciled family keys and enforce data rights,
external holdout reservations and applicable chemistry. This creates a linear
head, not a qualified HSP model or calibrated uncertainty.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

from .models import LinearHead

DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit_group_selected_ridge(x, y, groups, *, alphas=DEFAULT_ALPHAS, folds=5, seed=42):
    """Select each target by family-macro validation MAE, including a constant.

    Standardization and candidate fitting are repeated inside each training
    fold. In a tie prefer the constant, then the largest ridge penalty. The
    returned head is fitted on all supplied rows, which must be training rows.
    This selection score is not an independent test of the selected model.
    """
    try:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Features and targets must be numeric matrices") from None
    group = np.asarray(groups)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[0] != y.shape[0]
        or x.shape[1] == 0
        or y.shape[1] == 0
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or group.ndim != 1
        or len(group) != len(x)
        or any(
            not ((isinstance(g, str) and g.strip()) or isinstance(g, (int, np.integer)))
            or isinstance(g, (bool, np.bool_))
            for g in group
        )
    ):
        raise ValueError("Require finite aligned matrices and string or integer family keys")
    if len(set(group)) < 3:
        raise ValueError("Selection needs at least three training families")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least two")
    try:
        penalties = sorted(set(float(alpha) for alpha in alphas), reverse=True)
    except (TypeError, ValueError):
        raise ValueError("alphas must be finite positive penalties") from None
    if not penalties or len(penalties) > 32 or any(not np.isfinite(a) or a <= 0 for a in penalties):
        raise ValueError("Provide one to 32 finite positive penalties")
    candidates = [None, *penalties]
    oof = np.empty((len(candidates), *y.shape))
    seen = np.zeros(len(x), dtype=int)
    partitions = []
    splitter = GroupKFold(n_splits=min(folds, len(set(group))), shuffle=True, random_state=seed)
    for train, validation in splitter.split(x, groups=group):
        assert set(group[train]).isdisjoint(group[validation])
        oof[0, validation] = y[train].mean(axis=0)
        for index, alpha in enumerate(candidates[1:], start=1):
            oof[index, validation] = LinearHead.fit(x[train], y[train], alpha).predict(
                x[validation]
            )
        seen[validation] += 1
        partitions.append(
            {"train_indices": train.tolist(), "validation_indices": validation.tolist()}
        )
    if not np.all(seen == 1) or not np.isfinite(oof).all():
        raise ValueError("Cross-validation did not produce finite predictions exactly once per row")
    errors = np.abs(oof - y)
    scores = np.mean([errors[:, group == key].mean(axis=1) for key in sorted(set(group))], axis=0)
    selected = [int(np.flatnonzero(column <= column.min() + 1e-12)[0]) for column in scores.T]
    base = LinearHead.fit(x, y, alpha=1.0)
    coef, intercept = base.coef.copy(), base.intercept.copy()
    fitted = {}
    for target, choice in enumerate(selected):
        alpha = candidates[choice]
        if alpha is None:
            coef[target] = 0
            intercept[target] = y[:, target].mean()
        else:
            if alpha not in fitted:
                fitted[alpha] = LinearHead.fit(x, y, alpha)
            coef[target], intercept[target] = (
                fitted[alpha].coef[target],
                fitted[alpha].intercept[target],
            )
    head = LinearHead(base.mean, base.scale, coef, intercept)
    if not np.isfinite(head.predict(x)).all():
        raise ValueError("Selected training head produced nonfinite predictions")
    report = {
        "selection_scope": "supplied_training_rows_only",
        "criterion": "family_macro_validation_mae_per_target",
        "candidates": candidates,
        "candidate_scores": scores.tolist(),
        "selected_alphas": [candidates[index] for index in selected],
        "constant_component_alphas_are_null": True,
        "tie_rule": "within 1e-12 prefer constant, then larger alpha",
        "training_rows": len(x),
        "training_families": len(set(group)),
        "folds": partitions,
        "fold_standardization": "training_only",
        "independent_accuracy_validated": False,
        "calibrated_intervals": False,
    }
    return head, report
