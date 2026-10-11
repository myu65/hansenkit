"""Descriptive array-only fixed-charge dipole fluctuation diagnostics.

SI conducting-boundary population plug-in from Beauchamp2015 Eq3/Fingerhut2021
Eq44-45. The finite-window result is descriptive and is never a qualified
experimental dielectric constant or a Hansen parameter. All supplied frames
remain in the sample; no equilibration interval or parameter is selected here.
"""

import math
from numbers import Real

import numpy as np

from hansenkit.md_diagnostics import diagnose_trace

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_M = 8.8541878188e-12
VACUUM_PERMITTIVITY_STANDARD_UNCERTAINTY_F_M = 1.4e-21
NM_TO_M = 1e-9
NM3_TO_M3 = 1e-27
ASSUMPTIONS = (
    "Caller must establish neutral molecular dipoles, source units and consistent periodic gauge.",
    "Conducting boundaries and a compatible constant configured temperature ensemble "
    "are assumed, not verified.",
    "The denominator uses ensemble mean volume; NPT compatibility and isotropy "
    "are caller responsibilities.",
    "Population moments are finite-window plug-ins without correlation or finite-size "
    "bias correction.",
    "Fixed-charge electronic baseline1 excludes induced/electronic polarization "
    "and optical correction.",
    "ACF/ESS and block errors additionally require the assumptions recorded by md_diagnostics.",
    "Blocks/halves have independently centered covariance; short windows can suppress "
    "slow orientational variance.",
    "No equilibrium, material scope, physical accuracy, coverage or training permission "
    "follows from finite arithmetic.",
)


def _inputs(m_e_nm, volume_nm3, temperature_k):
    m = np.asarray(m_e_nm)
    volume = np.asarray(volume_nm3)
    if m.dtype.kind not in "fiu" or volume.dtype.kind not in "fiu":
        raise ValueError("Real numeric dipoles and volumes required")
    m, volume = m.astype(np.float64), volume.astype(np.float64)
    if (
        m.ndim != 2
        or m.shape[1] != 3
        or len(m) < 2
        or volume.shape != (len(m),)
        or not np.isfinite(m).all()
        or not np.isfinite(volume).all()
        or np.any(volume <= 0)
    ):
        raise ValueError("Aligned finite vector M and positive nm3 volumes required")
    if (
        isinstance(temperature_k, (bool, np.bool_))
        or not isinstance(temperature_k, Real)
        or not math.isfinite(temperature_k)
        or temperature_k <= 0
    ):
        raise ValueError("Finite positive configured Kelvin temperature required")
    return m, volume, float(temperature_k)


def fluctuation_moments(m_e_nm, volume_nm3, *, configured_temperature_k):
    """Full vector covariance / ensemble-mean volume, not mean(frame ratio).

    Covariance denominator is n, matching a sample mean of centered outer
    products. It is an uncorrected finite-window plug-in; no n/(n-1) or finite
    correlation correction is substituted. Molecular source neutrality,
    geometry, charge provenance, boundary conditions and ensemble must be
    established by a separately reviewed adapter before real traces are used.
    """
    m, volume, temperature = _inputs(m_e_nm, volume_nm3, configured_temperature_k)
    mean = np.mean(m, axis=0)
    centered = m - mean
    covariance = centered.T @ centered / len(m)
    mean_volume = float(np.mean(volume))
    mean_volume_si = mean_volume * NM3_TO_M3
    trace = float(np.trace(covariance))
    mean_squared = float(mean @ mean)
    denominator = 3 * VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K * temperature * NM3_TO_M3
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
        or not all(
            math.isfinite(x)
            for x in (mean_volume, mean_volume_si, trace, mean_squared, denominator)
        )
        or mean_volume <= 0
        or mean_volume_si <= 0
        or trace < 0
        or denominator <= 0
    ):
        raise ValueError("Observable scale prevents finite positive SI aggregate moments")
    factor = (ELEMENTARY_CHARGE_C * NM_TO_M) ** 2 / denominator
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Configured scale prevents finite positive SI conversion factor")
    response = 1 + factor * trace / mean_volume
    if not np.isfinite(covariance).all() or not math.isfinite(response):
        raise ValueError("Observable magnitude prevents finite fluctuation moments")
    return {
        "sample_count": len(m),
        "mean_vector_e_nm": mean.tolist(),
        "mean_vector_c_m": (mean * ELEMENTARY_CHARGE_C * NM_TO_M).tolist(),
        "squared_mean_vector_e2_nm2": mean_squared,
        "population_centered_covariance_e2_nm2": covariance.tolist(),
        "population_centered_covariance_c2_m2": (
            covariance * (ELEMENTARY_CHARGE_C * NM_TO_M) ** 2
        ).tolist(),
        "covariance_trace_e2_nm2": trace,
        "mean_volume_nm3": mean_volume,
        "mean_volume_m3": mean_volume_si,
        "configured_temperature_k": temperature,
        "descriptive_fixed_charge_response": response,
        "response_kind": "finite_window_fixed_charge_conducting_boundary_plugin",
        "electronic_baseline": 1,
        "electronic_or_induced_polarization_included": False,
        "experimental_dielectric_prediction": None,
        "hsp_predictions": None,
        "equilibrium_qualified": False,
        "equilibrium_qualification": "not_assessed",
        "physical_scope_qualified": False,
        "ensemble_qualified": False,
        "assumptions": list(ASSUMPTIONS),
        "physical_accuracy_validated": False,
        "uncertainty_coverage_validated": False,
        "training_allowed": False,
        "covariance_denominator": "n; uncorrected population plug-in",
        "volume_convention": "ensemble mean V in denominator, no frame division",
    }


def diagnose_dipole_trace(time_ps, m_e_nm, volume_nm3, *, configured_temperature_k, block_count=16):
    """Retain whole traces and report orientational/covariance correlations.

    The influence series is the first-order centered ratio influence for
    trace(Cov M)/mean(V). Its ACF/ESS is conditional and cannot qualify the
    finite-window response or an interval. The fixed blocks use their own
    vector means; short-block covariance bias is retained and disclosed.
    """
    m, volume, temperature = _inputs(m_e_nm, volume_nm3, configured_temperature_k)
    full = fluctuation_moments(m, volume, configured_temperature_k=temperature)
    centered = m - np.asarray(full["mean_vector_e_nm"])
    centered_square = np.einsum("ij,ij->i", centered, centered)
    c = full["covariance_trace_e2_nm2"]
    v = full["mean_volume_nm3"]
    influence = ((centered_square - c) - (c / v) * (volume - v)) / v
    observables = {
        "M_x_e_nm": m[:, 0],
        "M_y_e_nm": m[:, 1],
        "M_z_e_nm": m[:, 2],
        "M_squared_e2_nm2": np.einsum("ij,ij->i", m, m),
        "centered_M_squared_e2_nm2": centered_square,
        "volume_nm3": volume,
        "covariance_over_volume_ratio_influence_e2_per_nm": influence,
    }
    # Retain all six unique tensor products: trace alone cannot show anisotropy.
    for a, axis_a in enumerate("xyz"):
        for b, axis_b in enumerate("xyz"):
            if b >= a:
                observables[f"centered_M_{axis_a}_M_{axis_b}_e2_nm2"] = (
                    centered[:, a] * centered[:, b]
                )
    diagnostics = diagnose_trace(time_ps, observables, block_count=block_count)
    blocks = [
        fluctuation_moments(m[a:b], volume[a:b], configured_temperature_k=temperature)
        for a, b in (
            (record["sample_start"], record["sample_stop_exclusive"])
            for record in diagnostics["blocks"]
        )
    ]
    half = len(m) // 2
    halves = [
        fluctuation_moments(m[a:b], volume[a:b], configured_temperature_k=temperature)
        for a, b in ((0, half), (half, len(m)))
    ]
    return {
        "whole_trace_moments": full,
        "whole_trace_diagnostics": diagnostics,
        "fixed_equal_blocks_moments": blocks,
        "first_and_second_half_moments": halves,
        "half_response_difference": (
            halves[1]["descriptive_fixed_charge_response"]
            - halves[0]["descriptive_fixed_charge_response"]
        ),
        "blocks_choose_no_burn_in_or_cutoff": True,
        "all_supplied_frames_retained": True,
        "block_covariance_bias_corrected": False,
        "experimental_dielectric_comparison": None,
        "hsp_predictions": None,
        "equilibrium_qualified": False,
        "equilibrium_qualification": "not_assessed",
        "physical_scope_qualified": False,
        "ensemble_qualified": False,
        "assumptions": list(ASSUMPTIONS),
        "physical_accuracy_validated": False,
        "uncertainty_coverage_validated": False,
        "training_allowed": False,
    }
