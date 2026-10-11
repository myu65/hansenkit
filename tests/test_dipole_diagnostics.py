"""Authored array fixtures; no coordinates, force-field parameters or measured labels."""

import json
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from hansenkit.dipole_diagnostics import diagnose_dipole_trace, fluctuation_moments


def response_scale(temperature):
    with localcontext() as context:
        context.prec = 60
        return Decimal("1.602176634e-28") ** 2 / (
            3
            * Decimal("8.8541878188e-12")
            * Decimal("1.380649e-23")
            * Decimal(str(temperature))
            * Decimal("1e-27")
        )


def test_population_covariance_and_SI_response_independent_fraction_decimal_oracle():
    vectors = [[1, 2, -1], [3, -1, 2], [-2, 1, 4], [0, -2, -3]]
    volumes = [1, 2, 4, 9]
    mean = [sum(Fraction(row[a]) for row in vectors) / 4 for a in range(3)]
    covariance = [
        [
            sum((Fraction(row[a]) - mean[a]) * (Fraction(row[b]) - mean[b]) for row in vectors) / 4
            for b in range(3)
        ]
        for a in range(3)
    ]
    trace = sum(covariance[a][a] for a in range(3))
    expected = 1 + response_scale(310) * Decimal(trace.numerator) / Decimal(trace.denominator) / 4
    result = fluctuation_moments(vectors, volumes, configured_temperature_k=310)
    np.testing.assert_allclose(
        result["population_centered_covariance_e2_nm2"],
        [[float(v) for v in row] for row in covariance],
        rtol=1e-14,
    )
    assert result["descriptive_fixed_charge_response"] == pytest.approx(float(expected), rel=2e-15)
    np.testing.assert_allclose(
        result["mean_vector_c_m"],
        np.array(mean, dtype=float) * 1.602176634e-28,
        rtol=2e-15,
        atol=0,
    )
    np.testing.assert_allclose(
        result["population_centered_covariance_c2_m2"],
        np.array(covariance, dtype=float) * 1.602176634e-28**2,
        rtol=2e-15,
        atol=0,
    )
    assert result["mean_volume_m3"] == pytest.approx(4e-27, rel=2e-15, abs=0)
    assert result["covariance_denominator"].startswith("n;")


def test_vector_centering_constant_offset_and_rotation_invariance():
    vectors = np.array([[1, -2, 0], [-1, 2, 0], [3, 0, 1], [-3, 0, -1]], dtype=float)
    volumes = np.full(4, 5.0)
    first = fluctuation_moments(vectors, volumes, configured_temperature_k=320)
    shifted = fluctuation_moments(vectors + [100, -200, 300], volumes, configured_temperature_k=320)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = fluctuation_moments(vectors @ rotation.T, volumes, configured_temperature_k=320)
    cov = np.array(first["population_centered_covariance_e2_nm2"])
    np.testing.assert_allclose(shifted["population_centered_covariance_e2_nm2"], cov, atol=1e-14)
    np.testing.assert_allclose(
        rotated["population_centered_covariance_e2_nm2"], rotation @ cov @ rotation.T
    )
    assert (
        shifted["descriptive_fixed_charge_response"] == first["descriptive_fixed_charge_response"]
    )
    assert (
        rotated["descriptive_fixed_charge_response"] == first["descriptive_fixed_charge_response"]
    )
    opposite = fluctuation_moments([[1, 0, 0], [-1, 0, 0]], [5, 5], configured_temperature_k=320)
    assert opposite["covariance_trace_e2_nm2"] == 1  # Equal magnitudes still fluctuate as vectors.
    constant = fluctuation_moments([[7, -9, 4]] * 4, [5] * 4, configured_temperature_k=320)
    assert constant["descriptive_fixed_charge_response"] == 1
    assert constant["equilibrium_qualification"] == "not_assessed"


def test_NPT_mean_volume_and_configured_temperature_not_per_frame_ratio():
    m = np.array([[-1, 0, 0], [1, 0, 0], [-2, 0, 0], [2, 0, 0]])
    volume = np.array([1, 1, 9, 9])
    result = fluctuation_moments(m, volume, configured_temperature_k=311)
    expected = 1 + float(response_scale(311)) * 0.5
    wrong_frame_ratio = 1 + float(response_scale(311)) * float(np.mean(m[:, 0] ** 2 / volume))
    assert result["descriptive_fixed_charge_response"] == pytest.approx(expected)
    assert abs(result["descriptive_fixed_charge_response"] - wrong_frame_ratio) > 1
    hotter = fluctuation_moments(m, volume, configured_temperature_k=622)
    assert hotter["descriptive_fixed_charge_response"] - 1 == pytest.approx((expected - 1) / 2)


def test_whole_13_series_blocks_halves_and_JSON_are_retained_without_mutation():
    n = 128
    rng = np.random.default_rng(1187)
    m = rng.normal(0, 0.03, (n, 3))
    volume = 3 + rng.uniform(0, 0.1, n)
    before = m.copy(), volume.copy()
    report = diagnose_dipole_trace(
        np.arange(n) * 0.25, m, volume, configured_temperature_k=315, block_count=16
    )
    diagnostics = report["whole_trace_diagnostics"]
    assert diagnostics["sample_count"] == n
    assert len(diagnostics["observables"]) == 13
    assert len(report["fixed_equal_blocks_moments"]) == 16
    assert sum(b["sample_count"] for b in report["fixed_equal_blocks_moments"]) == n
    assert [b["sample_count"] for b in report["first_and_second_half_moments"]] == [64, 64]
    assert diagnostics["observables"]["covariance_over_volume_ratio_influence_e2_per_nm"][
        "mean"
    ] == pytest.approx(0, abs=1e-17)
    np.testing.assert_array_equal(m, before[0])
    np.testing.assert_array_equal(volume, before[1])
    json.dumps(report, allow_nan=False)
    for key in (
        "equilibrium_qualified",
        "physical_scope_qualified",
        "ensemble_qualified",
        "physical_accuracy_validated",
        "uncertainty_coverage_validated",
        "training_allowed",
    ):
        assert report[key] is False
    assert report["hsp_predictions"] is None


def test_short_blocks_retain_slow_orientation_covariance_bias_and_unresolved_constants():
    phase = np.arange(256) * (2 * np.pi / 256)
    vectors = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(256)))
    report = diagnose_dipole_trace(
        np.arange(256), vectors, np.full(256, 4.0), configured_temperature_k=300, block_count=16
    )
    whole = report["whole_trace_moments"]["covariance_trace_e2_nm2"]
    blocks = [b["covariance_trace_e2_nm2"] for b in report["fixed_equal_blocks_moments"]]
    assert np.mean(blocks) < whole / 10
    assert report["block_covariance_bias_corrected"] is False
    assert (
        "blocks_short_relative_to_estimated_correlation"
        in report["whole_trace_diagnostics"]["observables"]["M_x_e_nm"]["warnings"]
    )
    assert (
        report["whole_trace_diagnostics"]["observables"]["M_z_e_nm"]["effective_sample_size"]
        is None
    )


@pytest.mark.parametrize(
    "vectors,volume,temperature",
    [
        ([[1e200, 0, 0], [-1e200, 0, 0]], [1, 1], 300),
        ([[1e200, 0, 0]] * 2, [1, 1], 300),
        ([[0, 0, 0]] * 2, [1e308, 1e308], 300),
        ([[0, 0, 0]] * 2, [1e-320, 1e-320], 300),
        ([[0, 0, 0]] * 2, [1, 1], 1e-300),
    ],
)
def test_finite_inputs_that_overflow_or_underflow_required_moments_refuse(
    vectors, volume, temperature
):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        with pytest.raises(ValueError):
            fluctuation_moments(vectors, volume, configured_temperature_k=temperature)


@pytest.mark.parametrize(
    "vectors,volume",
    [
        ([[1, 2, 3]], [1]),
        ([[1, 2], [3, 4]], [1, 1]),
        ([[0, 0, 0]] * 2, [1]),
        ([[0, 0, 0]] * 2, [1, 0]),
        ([[0, 0, 0]] * 2, [1, -1]),
        ([[0, 0, 0]] * 2, [1, float("inf")]),
        ([[0, float("nan"), 0]] * 2, [1, 1]),
        ([["1", "2", "3"]] * 2, [1, 1]),
        ([[True, False, True]] * 2, [1, 1]),
        ([[1j, 0, 0]] * 2, [1, 1]),
    ],
)
def test_aligned_real_finite_input_refusals(vectors, volume):
    with pytest.raises(ValueError):
        fluctuation_moments(vectors, volume, configured_temperature_k=300)


@pytest.mark.parametrize(
    "temperature", [False, np.bool_(True), "300", 1j, 0, -1, float("nan"), float("inf")]
)
def test_configured_temperature_must_be_explicit_finite_positive_real(temperature):
    with pytest.raises(ValueError):
        fluctuation_moments([[0, 0, 0]] * 2, [1, 1], configured_temperature_k=temperature)


@pytest.mark.parametrize(
    "time,block_count",
    [
        (np.arange(31), 4),
        (np.arange(32), 16),
        (np.arange(65), 4),
        (np.arange(64), 3),
        (np.zeros(64), 4),
        (np.arange(64) ** 2, 4),
    ],
)
def test_trace_rejects_short_or_nonuniform_windows_without_cuts(time, block_count):
    with pytest.raises(ValueError):
        diagnose_dipole_trace(
            time,
            np.zeros((len(time), 3)),
            np.ones(len(time)),
            configured_temperature_k=300,
            block_count=block_count,
        )
