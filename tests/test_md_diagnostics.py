"""Original synthetic traces exercise statistical limitations, not real MD accuracy."""

import json

import numpy as np
import pytest

from hansenkit.md_diagnostics import diagnose_replicates, diagnose_trace


def test_iid_like_trace_has_near_sample_ess_and_honest_mean_uncertainty():
    n = 4096
    values = np.random.default_rng(627).normal(10, 2, n)
    result = diagnose_trace(np.arange(n) * 0.2, {"temperature": values}, block_count=8)
    summary = result["observables"]["temperature"]
    assert 0.6 * n < summary["effective_sample_size"] <= n
    assert summary["mean"] == pytest.approx(10, abs=0.15)
    assert summary["approximate_acf_mean_standard_error"] == pytest.approx(2 / np.sqrt(n), rel=0.35)
    assert summary["approximate_block_mean_standard_error"] > 0
    assert "large_descriptive_drift" not in summary["warnings"]
    assert result["equilibrium_qualification"] == "not_assessed"
    assert not result["physical_accuracy_validated"] and result["hsp_predictions"] is None
    json.dumps(result, allow_nan=False)


def test_long_autocorrelation_reduces_ess_and_increases_mean_error():
    n, phi = 8192, 0.95
    rng = np.random.default_rng(910)
    values = np.zeros(n + 2000)
    for index in range(1, len(values)):
        values[index] = phi * values[index - 1] + np.sqrt(1 - phi**2) * rng.normal()
    values = values[2000:]
    summary = diagnose_trace(np.arange(n), {"density": values}, block_count=8)["observables"][
        "density"
    ]
    theoretical_g = (1 + phi) / (1 - phi)
    assert 0.5 * theoretical_g < summary["statistical_inefficiency"] < 2 * theoretical_g
    assert summary["effective_sample_size"] < n / 20
    assert summary["approximate_acf_mean_standard_error"] > 4 / np.sqrt(n)
    assert summary["acf"][1] > 0.9


def test_drifting_trace_reports_change_and_does_not_declare_equilibrium():
    n = 1024
    values = np.linspace(0, 10, n) + np.random.default_rng(3).normal(0, 0.05, n)
    result = diagnose_trace(np.arange(n) * 0.01, {"energy": values})
    summary = result["observables"]["energy"]
    assert summary["half_mean_shift"] == pytest.approx(5, abs=0.1)
    assert summary["linear_trend_change_over_span"] == pytest.approx(10, abs=0.1)
    assert "large_descriptive_drift" in summary["warnings"]
    assert "low_estimated_effective_sample_size" in summary["warnings"]
    assert "blocks_short_relative_to_estimated_correlation" in summary["warnings"]
    assert result["equilibrium_qualification"] == "not_assessed"


@pytest.mark.parametrize("values", [np.full(64, 300.0), np.zeros(64)])
def test_constant_trace_does_not_get_zero_error_or_full_ess(values):
    summary = diagnose_trace(np.arange(64), {"temperature": values})["observables"]["temperature"]
    assert summary["status"] == "constant_or_numerically_unresolved"
    assert summary["effective_sample_size"] is None
    assert summary["statistical_inefficiency"] is None
    assert summary["approximate_block_mean_standard_error"] is None
    assert summary["approximate_acf_mean_standard_error"] is None
    assert "variance_unestimable_not_evidence_of_mixing" in summary["warnings"]
    json.dumps(summary, allow_nan=False)


def test_anticorrelation_never_inflates_effective_sample_count():
    n = 64
    summary = diagnose_trace(np.arange(n), {"alternating": (-1.0) ** np.arange(n)})["observables"][
        "alternating"
    ]
    assert summary["acf"][1] < -0.9
    assert summary["effective_sample_size"] <= n
    assert summary["statistical_inefficiency"] >= 1
    assert summary["ess_bound_applied"]


def test_block_boundaries_retain_every_sample_and_scale_time_units():
    n = 64
    values = np.random.default_rng(44).normal(size=n)
    first = diagnose_trace(5 + np.arange(n) * 0.2, {"density": values})
    scaled = diagnose_trace((5 + np.arange(n) * 0.2) * 1000, {"density": values})
    assert first["blocks"][0]["sample_start"] == 0
    assert first["blocks"][-1]["sample_stop_exclusive"] == n
    assert first["block_size_samples"] == 16
    assert first["block_duration"] == pytest.approx(3.2)
    assert scaled["sample_interval"] == pytest.approx(first["sample_interval"] * 1000)
    for key in ["mean", "half_mean_shift", "statistical_inefficiency", "effective_sample_size"]:
        assert scaled["observables"]["density"][key] == pytest.approx(
            first["observables"]["density"][key]
        )


def test_replicates_remain_separate_and_disagreement_is_visible():
    rng = np.random.default_rng(991)
    first, second = rng.normal(0, 1, 1024), rng.normal(10, 1, 2048)
    result = diagnose_replicates(
        {
            "seed_1": (np.arange(1024) * 0.1, {"density": first}),
            "seed_2": (np.arange(2048) * 0.2, {"density": second}),
        }
    )
    assert result["replicates"]["seed_1"]["sample_count"] == 1024
    assert result["replicates"]["seed_2"]["sample_count"] == 2048
    assert (
        result["replicates"]["seed_1"]["observables"]["density"]
        == diagnose_trace(np.arange(1024) * 0.1, {"density": first})["observables"]["density"]
    )
    assert result["between_replicate"]["density"]["between_replicate_mean_variance"] > 40
    assert result["between_replicate"]["density"][
        "equal_weight_mean_of_replicate_means"
    ] == pytest.approx(5, abs=0.15)
    assert "effective_sample_size" not in result
    assert result["replicate_independence"] == "caller_asserted_not_verified"
    assert result["equilibrium_qualification"] == "not_assessed"
    assert not result["physical_accuracy_validated"]
    json.dumps(result, allow_nan=False)


def test_one_replicate_cannot_estimate_between_run_uncertainty():
    result = diagnose_replicates(
        {"only": (np.arange(64), {"density": np.random.default_rng(14).normal(size=64)})}
    )
    spread = result["between_replicate"]["density"]
    assert spread["between_replicate_mean_variance"] is None
    assert spread["approximate_between_replicate_standard_error"] is None


def test_identically_constant_replicates_do_not_imply_zero_physical_uncertainty():
    result = diagnose_replicates(
        {
            "one": (np.arange(64), {"density": np.full(64, 0.8)}),
            "two": (np.arange(64), {"density": np.full(64, 0.8)}),
        }
    )
    spread = result["between_replicate"]["density"]
    assert spread["between_replicate_mean_variance"] == 0
    assert spread["approximate_between_replicate_standard_error"] is None
    assert "replicate_variance_unestimable" in spread["warnings"]
    assert not result["physical_accuracy_validated"]


@pytest.mark.parametrize(
    "time,observables,block_count",
    [
        (np.arange(16), {"x": np.ones(16)}, 4),
        (np.arange(33), {"x": np.ones(33)}, 4),
        (np.arange(64), {"x": np.ones(64)}, 16),
        (np.arange(64), {"x": np.ones(64)}, True),
        (np.arange(64), {"x": np.ones(64)}, 2),
        (np.zeros(64), {"x": np.ones(64)}, 4),
        (-np.arange(64), {"x": np.ones(64)}, 4),
        (np.r_[np.arange(63), 62.5], {"x": np.ones(64)}, 4),
        (np.r_[np.arange(63), np.nan], {"x": np.ones(64)}, 4),
        (np.arange(64), {"x": np.r_[np.ones(63), np.inf]}, 4),
        (np.arange(64), {"x": np.ones(63)}, 4),
        (np.arange(64), {"x": np.ones((64, 1))}, 4),
        (np.arange(64), {"x": np.full(64, "1")}, 4),
        (np.arange(64), {"": np.ones(64)}, 4),
        (np.arange(64), {}, 4),
        (np.arange(64, dtype=complex), {"x": np.ones(64)}, 4),
    ],
)
def test_malformed_sampling_or_observables_refused(time, observables, block_count):
    with pytest.raises(ValueError):
        diagnose_trace(time, observables, block_count=block_count)


@pytest.mark.parametrize(
    "replicates",
    [
        {},
        {"": (np.arange(64), {"x": np.ones(64)})},
        {"one": np.ones(64)},
        {
            "one": (np.arange(64), {"x": np.ones(64)}),
            "two": (np.arange(64), {"other": np.ones(64)}),
        },
    ],
)
def test_malformed_replicates_refused(replicates):
    with pytest.raises(ValueError):
        diagnose_replicates(replicates)
