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


def _hybrid_fold_candidates(x, embedding, y, groups, train, validation, base, candidates):
    """Rebuild residual targets inside an outer training partition."""
    if set(groups[train]) & set(groups[validation]):
        raise ValueError("Outer hybrid family leakage")
    head, selection = base(train)
    prediction = head.predict(x[validation])
    inner_oof = np.full_like(y[train], np.nan)
    inner_records = []
    splitter = GroupKFold(n_splits=min(4, len(set(groups[train]))), shuffle=True, random_state=42)
    for local_train, local_validation in splitter.split(x[train], groups=groups[train]):
        ids, held = train[local_train], train[local_validation]
        if set(groups[ids]) & (set(groups[held]) | set(groups[validation])):
            raise ValueError("Hybrid residual family leakage")
        inner_head, _ = base(ids)
        inner_oof[local_validation] = inner_head.predict(x[held])
        inner_records.append({"train_indices": ids.tolist(), "validation_indices": held.tolist()})
    if not np.isfinite(inner_oof).all():
        raise ValueError("Incomplete inner residual predictions")
    residual_y = y[train] - inner_oof
    outputs = [prediction, prediction + residual_y.mean(axis=0)]
    outputs.extend(
        prediction
        + LinearHead.fit(embedding[train], residual_y, alpha).predict(embedding[validation])
        for alpha in candidates[2:]
    )
    return (
        np.array(outputs),
        prediction,
        {
            "train_indices": train.tolist(),
            "validation_indices": validation.tolist(),
            "inner_residual_folds": inner_records,
            "base_selection": selection,
        },
    )


def fit_group_selected_hybrid(x, embedding, y, groups, *, alphas=DEFAULT_ALPHAS, seed=42):
    """Fit chemical base plus embedding residual using supplied training families.

    Five outer family folds select residual strength; four inner family folds
    reconstruct its targets. Each base selects its penalty using only its own
    training subset. Caller owns rights, chemistry and external reservations.
    Returns two LinearHeads with JSON states and a training selection report.
    No independent accuracy, calibration or applicability claim is conferred.
    """
    try:
        x, embedding, y = (np.asarray(a, dtype=float) for a in (x, embedding, y))
    except (TypeError, ValueError):
        raise ValueError("Hybrid inputs must be finite aligned matrices") from None
    groups = np.asarray(groups)
    if (
        x.ndim != 2
        or embedding.ndim != 2
        or y.ndim != 2
        or len(x) != len(embedding)
        or len(x) != len(y)
        or not embedding.shape[1]
        or not np.isfinite(embedding).all()
        or groups.ndim != 1
        or len(groups) != len(x)
    ):
        raise ValueError("Hybrid inputs must be finite aligned matrices")
    if len(set(groups)) < 5:
        raise ValueError("Nested hybrid selection needs at least five training families")
    # Existing validation also checks features, targets, family keys and penalties.
    base_head, base_report = fit_group_selected_ridge(x, y, groups, alphas=alphas, seed=seed)
    candidates = ("none", "constant", *base_report["candidates"][1:])
    cache = {tuple(range(len(x))): (base_head, base_report)}

    def base(ids):
        key = tuple(int(i) for i in ids)
        if key not in cache:
            cache[key] = fit_group_selected_ridge(
                x[list(key)], y[list(key)], groups[list(key)], alphas=alphas, seed=seed
            )
        return cache[key]

    oof = np.full((len(candidates), *y.shape), np.nan)
    base_oof = np.full_like(y, np.nan)
    records = []
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, validation in splitter.split(x, groups=groups):
        candidate_predictions, prediction, record = _hybrid_fold_candidates(
            x, embedding, y, groups, train, validation, base, candidates
        )
        oof[:, validation], base_oof[validation] = candidate_predictions, prediction
        records.append(record)
    if not np.isfinite(oof).all() or not np.isfinite(base_oof).all():
        raise ValueError("Incomplete outer hybrid predictions")
    errors = np.abs(oof - y)
    scores = np.mean([errors[:, groups == key].mean(axis=1) for key in sorted(set(groups))], axis=0)
    choices = [int(np.flatnonzero(s <= s.min() + 1e-12)[0]) for s in scores.T]
    residual_y = y - base_oof
    residual = LinearHead.fit(embedding, residual_y, 1.0)
    fitted = {}
    for target, choice in enumerate(choices):
        candidate = candidates[choice]
        if candidate in ("none", "constant"):
            residual.coef[target] = 0
            residual.intercept[target] = (
                0.0 if candidate == "none" else residual_y[:, target].mean()
            )
        else:
            if candidate not in fitted:
                fitted[candidate] = LinearHead.fit(embedding, residual_y, candidate)
            residual.coef[target] = fitted[candidate].coef[target]
            residual.intercept[target] = fitted[candidate].intercept[target]
    report = {
        "selection_scope": "supplied_training_rows_only",
        "training_rows": len(x),
        "training_families": len(set(groups)),
        "base_selection": base_report,
        "residual_candidates": list(candidates),
        "candidate_scores": scores.tolist(),
        "selected_candidates": [candidates[i] for i in choices],
        "outer_folds": records,
        "inner_folds_seed": 42,
        "outer_folds_seed": seed,
        "base_subset_selections": [
            {"train_indices": list(ids), "selection": selection}
            for ids, (_, selection) in sorted(cache.items())
        ],
        "tie_rule": "within 1e-12 prefer no residual, then constant, then larger alpha",
        "criterion": "family_macro_validation_mae_per_target",
        "independent_accuracy_validated": False,
        "calibrated_intervals": False,
    }
    return base_head, residual, report
