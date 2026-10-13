"""Original synthetic arrays and two-port strings, never reference tables/weights."""

import copy
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import hansenkit.scalar_reference as m
from hansenkit.chemistry import normalize_smiles
from hansenkit.schema import PolymerInput
from hansenkit.scope import assess_scope


@pytest.fixture(autouse=True)
def limited_threads():
    with threadpool_limits(limits=4):
        yield


def provenance(task="single_delta_total"):
    result = {
        "task": task,
        "source_unit": m.UNITS[task],
        "label_kind": "synthetic",
        "audit_basis": "operator_assumption",
        "use_scope": "scientific_private_noncommercial",
        "allowed_role": "authored_fixture_only",
        "authored_fixture_only": True,
        "rights_status": "approved",
        "training_allowed": True,
        "private_derived_weights_allowed": True,
        **dict.fromkeys(m.FALSE_FLAGS, False),
        "HSP_predictions": None,
        **dict.fromkeys(
            (
                "source_asset_sha256",
                "whole_source_name_bridge_sha256",
                "reservation_sha256",
                "structure_split_sha256",
                "fitting_review_sha256",
            ),
            "a" * 64,
        ),
    }
    if task == "density":
        result.update(
            source_unit_review_status="approved",
            source_unit_review_sha256="b" * 64,
            whole_candidate_inventory_review_status="approved_local_known_candidates_only",
            whole_candidate_inventory_review_sha256="c" * 64,
            global_isolation_certificate_qualified=False,
            target_count_kind="density_synthetic_only",
            HSP_or_QM_training_count_increment=0,
        )
    return result


def population():
    rng = np.random.default_rng(771)
    x = rng.normal(size=(18, 19))
    x[:, 18] = 3.0  # inactive but still part of raw OOD domain
    y = np.exp(0.3 + 0.18 * x[:, 0] - 0.08 * x[:, 1])
    groups = np.repeat(np.arange(6), [1, 2, 3, 4, 5, 3])
    return x, y, groups


@pytest.mark.parametrize("kind", ["ridge", "rbf"])
@pytest.mark.parametrize("task", ["single_delta_total", "CED", "density"])
def test_positive_roundtrip_source_units_and_copy_isolation(kind, task):
    x, y, groups = population()
    head = m.ScalarReferenceHead.fit(
        x, y, groups, kind=kind, alpha=1.0, provenance=provenance(task)
    )
    saved = head.state()
    restored = m.ScalarReferenceHead.restore(json.loads(json.dumps(saved, allow_nan=False)))
    np.testing.assert_array_equal(head.predict(x), restored.predict(x))
    assert (head.predict(x) > 0).all()
    assert saved["provenance"]["label_kind"] == "synthetic"
    assert (
        saved["provenance"]["task"] == task and saved["provenance"]["source_unit"] == m.UNITS[task]
    )
    assert saved["provenance"]["physical_accuracy_qualified"] is False
    assert saved["provenance"]["HSP_predictions"] is None
    saved["center"][0] += 99
    assert head.state()["center"][0] != saved["center"][0]


def test_weighted_Ridge_matches_independent_augmented_least_squares():
    x, y, groups = population()
    head = m.ScalarReferenceHead.fit(x, y, groups, kind="ridge", alpha=0.1, provenance=provenance())
    weights = np.array([1 / (6 * np.count_nonzero(groups == g)) for g in groups])
    state = head.state()
    active = np.asarray(state["mask"], bool)
    z = (x[:, active] - np.asarray(state["center"])[active]) / np.asarray(state["scale"])
    centered_y = np.log(y) - weights @ np.log(y)
    design = np.vstack((np.sqrt(weights)[:, None] * z, np.sqrt(0.1) * np.eye(z.shape[1])))
    response = np.r_[np.sqrt(weights) * centered_y, np.zeros(z.shape[1])]
    coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
    np.testing.assert_allclose(state["coefficient"], coefficient, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        head.predict(x), np.exp(weights @ np.log(y) + z @ coefficient), rtol=1e-12, atol=1e-12
    )
    for family in range(6):
        assert np.isclose(weights[groups == family].sum(), 1 / 6)


def test_kernel_centering_and_redigested_corruption_refuses():
    x, y, groups = population()
    head = m.ScalarReferenceHead.fit(x, y, groups, kind="rbf", alpha=1.0, provenance=provenance())
    saved = head.state()
    z, w = np.asarray(saved["training_coordinates"]), np.asarray(saved["weights"])
    distance = np.sum((z[:, None] - z[None, :]) ** 2, axis=2)
    kernel = np.exp(-distance / 19)
    means = kernel @ w
    centered = kernel - means[:, None] - means[None, :] + float(w @ means)
    np.testing.assert_allclose(centered @ w, 0, rtol=0, atol=1e-15)
    saved["kernel_training_row_mean"][0] += 0.2
    saved["state_digest"] = m._digest(saved)
    with pytest.raises(m.NumericalRefusal, match="centering"):
        m.ScalarReferenceHead.restore(saved)


def test_group_selection_rebuilds_training_only_and_held_poison_has_no_effect():
    x, y, groups = population()
    # Six training families plus separate outside rows; those values are never adopted.
    full_x = np.vstack((x, np.full((3, 19), 9.0)))
    full_y, full_groups = np.r_[y, [12.0, 13.0, 14.0]], np.r_[groups, [10, 11, 12]]
    indices = np.arange(len(x))
    first, report = m.fit_group_selected_scalar_reference(
        full_x, full_y, full_groups, training_indices=indices, provenance=provenance()
    )
    full_x[len(x) :] = np.nan
    full_y[len(x) :] = np.inf
    second, poisoned = m.fit_group_selected_scalar_reference(
        full_x, full_y, full_groups, training_indices=indices, provenance=provenance()
    )
    assert first.state() == second.state() and report == poisoned
    assert report["held_labels_or_features_used"] is False
    assert [(r["kind"], r["alpha"]) for r in report["candidates"]] == list(m.GRID)
    assert report["training_indices"] == indices.tolist()


def test_density_train_selected_reviews_counts_and_held_protocols_stay_separate():
    class Outside:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Density held array protocol executed")

        def __float__(self):
            pytest.fail("Density held target converted")

        def __str__(self):
            pytest.fail("Density held family converted")

    x, y, groups = population()
    selected, report = m.fit_group_selected_scalar_reference(
        x, y, groups, training_indices=np.arange(len(x)), provenance=provenance("density")
    )
    poisoned, repeated = m.fit_group_selected_scalar_reference(
        x.tolist() + [Outside()],
        y.tolist() + [Outside()],
        groups.tolist() + [Outside()],
        training_indices=np.arange(len(x)),
        provenance=provenance("density"),
    )
    assert selected.state() == poisoned.state() and report == repeated
    assert report["held_labels_or_features_used"] is False
    assert selected.state()["provenance"]["target_count_kind"] == "density_synthetic_only"
    assert selected.state()["provenance"]["HSP_or_QM_training_count_increment"] == 0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("source_unit", "g/cm3"),
        ("source_unit_review_status", "pending"),
        ("source_unit_review_sha256", None),
        ("whole_candidate_inventory_review_status", "approved"),
        ("whole_candidate_inventory_review_sha256", " "),
        ("global_isolation_certificate_qualified", True),
        ("global_isolation_certificate_qualified", 0),
        ("target_count_kind", "density_reference_auxiliary_only"),
        ("HSP_or_QM_training_count_increment", False),
        ("HSP_or_QM_training_count_increment", 1),
        ("allowed_role", "evaluation_only"),
        ("physical_accuracy_qualified", True),
        ("default_predictor_enabled", True),
    ],
)
def test_density_reviews_and_countkind_refuse_before_array_and_on_redigested_restore(field, bad):
    class Unread:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Density arrays converted before source review")

    declared = provenance("density")
    declared[field] = bad
    with pytest.raises(m.NumericalRefusal):
        m.ScalarReferenceHead.fit(
            Unread(), Unread(), Unread(), kind="ridge", alpha=1.0, provenance=declared
        )
    x, y, groups = population()
    saved = m.ScalarReferenceHead.fit(
        x, y, groups, kind="ridge", alpha=1.0, provenance=provenance("density")
    ).state()
    saved["provenance"][field] = bad
    saved["state_digest"] = m._digest(saved)
    with pytest.raises(m.NumericalRefusal):
        m.ScalarReferenceHead.restore(saved)


def test_density_reference_review_metadata_copy_and_private_state_namespace():
    x, y, groups = population()
    declared = provenance("density")
    declared.update(
        label_kind="published_reference_auxiliary",
        authored_fixture_only=False,
        allowed_role="source_specific_local_training",
        target_count_kind="density_reference_auxiliary_only",
    )
    head = m.ScalarReferenceHead.fit(x, y, groups, kind="ridge", alpha=1.0, provenance=declared)
    # These arrays are authored; only the metadata route simulates a reviewed reference caller.
    declared["source_unit_review_sha256"] = "d" * 64
    saved = head.state()
    assert saved["provenance"]["source_unit_review_sha256"] == "b" * 64
    assert saved["task"] == "scalar_reference_head"
    restored = m.ScalarReferenceHead.restore(json.loads(json.dumps(saved)))
    np.testing.assert_array_equal(head.predict(x), restored.predict(x))
    assert restored.state()["provenance"]["whole_candidate_inventory_review_status"] == (
        "approved_local_known_candidates_only"
    )
    assert restored.state()["provenance"]["global_isolation_certificate_qualified"] is False
    saved["task"] = "polybench_density_reference_head"
    saved["state_digest"] = m._digest(saved)
    with pytest.raises(m.NumericalRefusal, match="schema"):
        m.ScalarReferenceHead.restore(saved)


@pytest.mark.parametrize("container", ["list", "object_ndarray"])
def test_held_python_protocols_never_run_before_training_index_selection(container):
    class Outside:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Held __array__ executed")

        def __float__(self):
            pytest.fail("Held __float__ executed")

        def __str__(self):
            pytest.fail("Held __str__ executed")

    x, y, groups = population()
    baseline, report = m.fit_group_selected_scalar_reference(
        x, y, groups, training_indices=np.arange(len(x)), provenance=provenance()
    )
    outside = Outside()
    if container == "list":
        xx, yy, gg = x.tolist() + [outside], y.tolist() + [outside], groups.tolist() + [outside]
    else:
        xx, yy, gg = (
            np.empty((len(x) + 1, 19), object),
            np.empty(len(x) + 1, object),
            np.empty(len(x) + 1, object),
        )
        xx[: len(x)] = x
        yy[: len(x)] = y
        gg[: len(x)] = groups
        for column in range(19):
            xx[-1, column] = outside
        yy[-1], gg[-1] = outside, outside
    second, observed = m.fit_group_selected_scalar_reference(
        xx, yy, gg, training_indices=np.arange(len(x)), provenance=provenance()
    )
    assert baseline.state() == second.state() and report == observed


@pytest.mark.parametrize(
    "bad", ["", "  ", None, float("nan"), True, 1.0, [1], {"family": 1}, "integer_string_mixed"]
)
def test_family_keys_no_missing_boolean_nonscalar_or_implicit_namespace_merge(bad):
    x, y, groups = population()
    invalid = groups.tolist()
    invalid[0] = "1" if bad == "integer_string_mixed" else bad
    with pytest.raises(m.NumericalRefusal, match="[Ff]amily|namespace"):
        m.ScalarReferenceHead.fit(x, y, invalid, kind="ridge", alpha=1.0, provenance=provenance())
    valid = ["owned-family-" + str(group) for group in groups]
    head = m.ScalarReferenceHead.fit(x, y, valid, kind="ridge", alpha=1.0, provenance=provenance())
    assert (head.predict(x) > 0).all()


def test_all19_OOD_inactive_constants_and_training_refs_survive_extreme_query():
    x, y, groups = population()
    head = m.ScalarReferenceHead.fit(x, y, groups, kind="ridge", alpha=1.0, provenance=provenance())
    before = head.state()
    query = x[[0]].copy()
    query[:, 18] = 4.0
    report = head.ood(query)
    assert report["outside_training_range"] == [True] and report["outside_feature_indices"] == [
        [18]
    ]
    assert report["physical_or_calibrated_uncertainty_qualified"] is False
    assert before["mask"][18] is False
    query[:, 0] = 1e6
    head.ood(query)
    assert head.state()["train_min"] == before["train_min"]
    assert head.state()["train_max"] == before["train_max"]
    with pytest.raises(m.NumericalRefusal, match="bounds"):
        broken = copy.deepcopy(before)
        broken["train_min"][0] = broken["train_max"][0] + 1
        broken["state_digest"] = m._digest(broken)
        m.ScalarReferenceHead.restore(broken)


@pytest.mark.parametrize(
    "poison",
    [
        "training",
        "privateweights",
        "evaluation",
        "HSP",
        "units",
        "mixed",
        "teacher",
        "experimental",
    ],
)
def test_caller_rights_label_kind_and_unit_poison_refuse_before_array_conversion(poison):
    class Unread:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Array accessed before source permission")

    prov = provenance()
    if poison == "training":
        prov["training_allowed"] = 0
    elif poison == "privateweights":
        prov["private_derived_weights_allowed"] = False
    elif poison == "evaluation":
        prov["allowed_role"] = "evaluation_only"
    elif poison == "HSP":
        prov["HSP_backend_qualified"] = True
    elif poison == "units":
        prov["source_unit"] = m.UNITS["CED"]
    elif poison == "mixed":
        prov["label_kind"] = ["synthetic", "published_reference_auxiliary"]
    else:
        prov["label_kind"] = poison
    with pytest.raises(m.NumericalRefusal):
        m.ScalarReferenceHead.fit(
            Unread(), Unread(), Unread(), kind="ridge", alpha=1.0, provenance=prov
        )


def test_published_auxiliary_and_synthetic_provenance_never_collapse():
    x, y, groups = population()
    declared = provenance("CED")
    declared.update(
        label_kind="published_reference_auxiliary",
        authored_fixture_only=False,
        allowed_role="source_specific_local_training",
    )
    # Arbitrary authored arrays simulate this metadata path, never actual source observations.
    head = m.ScalarReferenceHead.fit(x, y, groups, kind="ridge", alpha=1.0, provenance=declared)
    assert head.state()["provenance"] == declared
    assert head.state()["provenance"]["physical_accuracy_qualified"] is False


def test_invalid_target_zero_underflow_and_shape_no_fallback():
    x, y, groups = population()
    for response in (np.zeros_like(y), -y, np.full(len(y), np.nan)):
        with pytest.raises(m.NumericalRefusal):
            m.ScalarReferenceHead.fit(
                x, response, groups, kind="ridge", alpha=1.0, provenance=provenance()
            )
    with pytest.raises(m.NumericalRefusal):
        m.ScalarReferenceHead.fit(
            x[:, :-1], y, groups, kind="ridge", alpha=1.0, provenance=provenance()
        )
    small = np.ones_like(x)
    small[:, 0] = np.linspace(0.0, 1e-200, len(x))
    with pytest.raises(m.NumericalRefusal, match="underflow"):
        m.ScalarReferenceHead.fit(
            small, y, groups, kind="ridge", alpha=1.0, provenance=provenance()
        )
    with pytest.raises(m.NumericalRefusal):
        m.fit_group_selected_scalar_reference(
            x, y, groups, training_indices=[-1], provenance=provenance()
        )


@pytest.mark.parametrize("bad", [{}, {"mask": []}, {"alpha": True}, {"schema_version": True}])
def test_malformed_restore_refuses(bad):
    if len(bad) == 1:
        x, y, groups = population()
        base = m.ScalarReferenceHead.fit(
            x, y, groups, kind="ridge", alpha=1.0, provenance=provenance()
        ).state()
        base.update(bad)
        base["state_digest"] = m._digest(base)
    else:
        base = bad
    with pytest.raises(m.NumericalRefusal):
        m.ScalarReferenceHead.restore(base)


def test_original_two_port_unit_indices_missing_ownership_and_other_bucket():
    values, audit = m.raw_repeat_unit_features("*CCO*")
    assert values.shape == (19,) and len(m.FEATURE_NAMES) == 19
    assert audit["original_source_SMILES_unchanged"] == "*CCO*"
    assert audit["explicit_original_ports"] == [0, 4]
    assert audit["functional_unassigned_real_atoms"] == [3]
    assert audit["functional_atom_coverage_complete"] is False
    assert audit["composition_atom_coverage_fraction"] == 1
    assert audit["functional_channel_used_for_fitting"] is False
    assert set(audit["functional_owners"]).isdisjoint(audit["explicit_original_ports"])
    _, silicon = m.raw_repeat_unit_features("*O[Si](C)(C)O*")
    assert silicon["composition_atom_coverage_fraction"] == 1
    assert silicon["functional_unassigned_real_atoms"]
    assert silicon["physical_or_material_identity_qualified"] is False


def test_same_composition_distinguished_by_original_unit_Hbond_proxies_no_caps():
    a, aa = m.raw_repeat_unit_features("*CC(=O)NC*")
    b, bb = m.raw_repeat_unit_features("*CN(C)OC*")
    np.testing.assert_array_equal(a[:16], b[:16])
    assert (a[16:] != b[16:]).all()
    for original, audit in (("*CC(=O)NC*", aa), ("*CN(C)OC*", bb)):
        assert audit["original_source_SMILES_unchanged"] == original
        assert len(audit["explicit_original_ports"]) == 2
        assert audit["caps_Mn_Mw_ends_or_grouping_key_feature_substitution"] is False
        assert audit["HSP_predictions"] is None


@pytest.mark.parametrize(
    "raw", ["CCO", "*CC*.*CC*", "*C[N+](C)(C)CC*", "[13*]CC*", "[13CH2](*)CC*", "*=CC*"]
)
def test_unsupported_raw_units_refuse_and_default_polymer_scope_stays_off(raw):
    with pytest.raises(ValueError):
        m.raw_repeat_unit_features(raw)
    assert normalize_smiles("*CCO*").count("*") == 2
    verdict = assess_scope(
        PolymerInput(
            series_id="authored_scope_fixture",
            mn_g_mol=1000.0,
            repeat_units=[{"smiles": "*CCO*", "mole_fraction": 1.0}],
        )
    )
    assert verdict.supported is False
