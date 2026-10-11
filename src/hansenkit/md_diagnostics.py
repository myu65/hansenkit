"""Descriptive, assumption-bound diagnostics of uniformly sampled MD observables.

No burn-in is selected, frames are not treated as independent, and these summaries
cannot establish equilibrium or physical accuracy. The paired autocorrelation
estimator follows the Geyer initial positive/monotone sequence equations described
in https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.
Their justification assumes stationary, reversible sampling with finite variance;
applicability to a particular MD observable/integrator remains the caller's task.
This is an original NumPy implementation, not Stan's multi-chain estimator.
"""

from collections.abc import Mapping

import numpy as np

DIAGNOSTICS_VERSION = "md-timeseries-descriptive-v1"
MIN_SAMPLES = 32
ASSUMPTIONS = (
    "Uniform time sampling; all supplied samples are retained without selecting burn-in.",
    "ACF/ESS and mean errors assume stationary sampling with finite variance.",
    "The positive/monotone pair estimator is justified for reversible sampling; "
    "its applicability to this MD process has not been established here.",
    "Block mean error additionally assumes blocks are long enough to be nearly independent.",
    "Replicate IDs are caller assertions of independent runs of the same material/conditions.",
    "Small drift and large estimated ESS do not establish equilibrium or physical accuracy.",
)


def _vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a one-dimensional real numeric sequence")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _autocorrelation(values):
    """Biased covariance ACF, zero padded to avoid circular correlations."""
    centered = values - np.mean(values)
    # Rescaling avoids overflow in the FFT covariance products.
    centered /= np.max(np.abs(centered))
    n = len(centered)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    covariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)[: n // 2 + 1]
    return covariance / covariance[0]


def _correlation_estimates(values):
    acf = _autocorrelation(values)
    pairs = []
    stopped_at_nonpositive_pair = False
    for index in range(0, len(acf) - 1, 2):
        pair = float(acf[index] + acf[index + 1])
        if pair <= 0:
            stopped_at_nonpositive_pair = True
            break
        # Initial monotone sequence after the initial positive sequence truncation.
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    raw_g = -1 + 2 * sum(pairs)
    # Conservative convention: do not award ESS > n for anticorrelation.
    g = float(np.clip(raw_g, 1, len(values)))
    return {
        "acf": [float(value) for value in acf],
        "acf_max_lag_samples": len(acf) - 1,
        "retained_pair_count": len(pairs),
        "positive_sequence_truncated": stopped_at_nonpositive_pair,
        "raw_statistical_inefficiency": float(raw_g),
        "statistical_inefficiency": g,
        "effective_sample_size": float(len(values) / g),
        "estimator": "biased_acf_initial_positive_monotone_pairs",
        "ess_bound_applied": raw_g < 1 or raw_g > len(values),
    }


def _observable(time, values, block_count):
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    block_size = n // block_count
    block_means = np.mean(values.reshape(block_count, block_size), axis=1)
    between_block_variance = float(np.var(block_means, ddof=1))
    half = n // 2
    first_mean = float(np.mean(values[:half]))
    last_mean = float(np.mean(values[-half:]))
    shift = last_mean - first_mean
    scaled_time = (time - time[0]) / (time[-1] - time[0])
    scaled_time -= np.mean(scaled_time)
    span_trend = float(np.dot(scaled_time, values - mean) / np.dot(scaled_time, scaled_time))
    if not np.all(np.isfinite([mean, sd, between_block_variance, shift, span_trend])):
        raise ValueError("Observable magnitude prevents finite diagnostic summaries")
    result = {
        "mean": mean,
        "sample_standard_deviation": sd,
        "block_means": [float(value) for value in block_means],
        "between_block_mean_variance": between_block_variance,
        "first_half_mean": first_mean,
        "last_half_mean": last_mean,
        "half_mean_shift": shift,
        "linear_trend_change_over_span": span_trend,
        "normalized_half_shift": None,
        "normalized_trend_change_over_span": None,
        "approximate_block_mean_standard_error": None,
        "approximate_acf_mean_standard_error": None,
        "acf": None,
        "statistical_inefficiency": None,
        "effective_sample_size": None,
        "warnings": [],
    }
    numerical_floor = 32 * np.finfo(float).eps * float(np.max(np.abs(values)))
    if sd <= numerical_floor:
        result["status"] = "constant_or_numerically_unresolved"
        result["warnings"].append("variance_unestimable_not_evidence_of_mixing")
        return result
    result["status"] = "descriptive_only"
    result.update(_correlation_estimates(values))
    result["normalized_half_shift"] = shift / sd
    result["normalized_trend_change_over_span"] = span_trend / sd
    result["approximate_block_mean_standard_error"] = float(
        np.sqrt(between_block_variance / block_count)
    )
    result["approximate_acf_mean_standard_error"] = float(
        sd / np.sqrt(result["effective_sample_size"])
    )
    # Predeclared exploratory flags, not hypothesis tests or physical acceptance gates.
    if abs(shift / sd) > 0.5 or abs(span_trend / sd) > 1:
        result["warnings"].append("large_descriptive_drift")
    if result["effective_sample_size"] < 20:
        result["warnings"].append("low_estimated_effective_sample_size")
    if block_size < 10 * result["statistical_inefficiency"]:
        result["warnings"].append("blocks_short_relative_to_estimated_correlation")
    if not result["positive_sequence_truncated"]:
        result["warnings"].append("positive_sequence_reaches_available_lag_window")
    return result


def diagnose_trace(time, observables, *, block_count=4):
    """Return JSON-safe summaries of one run; at least 32 aligned finite samples.

    ``observables`` maps nonempty names to real one-dimensional sequences. Time
    units are chosen by the caller and sampling must be increasing and uniform
    (relative tolerance 1e-6). The entire trace is divided into ``block_count``
    equal blocks of at least eight samples, requiring exact divisibility. No
    samples are silently discarded or selected using diagnostic results.

    Returned standard errors are approximate and conditional on the stated
    assumptions. Drift flags use 0.5 SD half-shift or 1 SD trend-over-span; low
    ESS uses 20 and short blocks use 10 times statistical inefficiency. These
    exploratory thresholds are recorded, not fitted or physical acceptance tests.
    """
    time = _vector(time, "time")
    n = len(time)
    if n < MIN_SAMPLES:
        raise ValueError(f"At least {MIN_SAMPLES} samples are required")
    if type(block_count) is not int or block_count < 4 or n % block_count:
        raise ValueError("At least four equal blocks with exact sample divisibility are required")
    block_size = n // block_count
    if block_size < 8:
        raise ValueError("Each block must contain at least eight samples")
    intervals = np.diff(time)
    if not np.all(np.isfinite(intervals)) or not np.all(intervals > 0):
        raise ValueError("Time must be finite and strictly increasing")
    interval = float(np.mean(intervals))
    if not np.allclose(intervals, interval, rtol=1e-6, atol=0):
        raise ValueError("Uniform time sampling is required")
    if not isinstance(observables, Mapping) or not observables:
        raise ValueError("Provide a nonempty mapping of named observables")
    diagnostics = {}
    for name, value in observables.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Observable names must be nonempty strings")
        values = _vector(value, name)
        if values.shape != time.shape:
            raise ValueError(f"Observable {name} must align with the time samples")
        diagnostics[name] = _observable(time, values, block_count)
    return {
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "sample_count": n,
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "sample_interval": interval,
        "block_count": block_count,
        "block_size_samples": block_size,
        "block_duration": block_size * interval,
        "blocks": [
            {
                "sample_start": index * block_size,
                "sample_stop_exclusive": (index + 1) * block_size,
                "time_start": float(time[index * block_size]),
                "time_stop_exclusive": float(time[(index + 1) * block_size - 1] + interval),
            }
            for index in range(block_count)
        ],
        "observables": diagnostics,
        "exploratory_thresholds": {
            "absolute_normalized_half_shift": 0.5,
            "absolute_normalized_trend_over_span": 1.0,
            "minimum_estimated_ess": 20,
            "minimum_block_size_over_inefficiency": 10,
        },
        "assumptions": list(ASSUMPTIONS),
        "equilibrium_qualification": "not_assessed",
        "physical_accuracy_validated": False,
        "hsp_predictions": None,
    }


def diagnose_replicates(replicates, *, block_count=4):
    """Diagnose runs separately; never join their endpoints or sum their ESS.

    ``replicates`` maps unique nonempty IDs to ``(time, observables)`` tuples.
    Runs may have different lengths and sampling intervals, but observable names
    must agree. Equal-weight between-replicate mean variance/error is descriptive
    and additionally assumes independent runs of the same material/conditions.
    One replicate cannot estimate between-replicate uncertainty (returns None).
    """
    if not isinstance(replicates, Mapping) or not replicates:
        raise ValueError("Provide a nonempty mapping of independent replicate IDs")
    reports = {}
    names = None
    for replicate_id, trace in replicates.items():
        if not isinstance(replicate_id, str) or not replicate_id.strip():
            raise ValueError("Replicate IDs must be nonempty strings")
        if not isinstance(trace, (tuple, list)) or len(trace) != 2:
            raise ValueError("Each replicate must be a (time, observables) pair")
        report = diagnose_trace(trace[0], trace[1], block_count=block_count)
        if names is not None and set(report["observables"]) != names:
            raise ValueError("Replicates must provide the same observable names")
        names = set(report["observables"])
        reports[replicate_id] = report
    spread = {}
    for name in sorted(names):
        means = np.array([report["observables"][name]["mean"] for report in reports.values()])
        variance = float(np.var(means, ddof=1)) if len(means) > 1 else None
        if variance is not None and not np.isfinite(variance):
            raise ValueError("Replicate mean spread prevents finite diagnostic summaries")
        unresolved = any(
            report["observables"][name]["status"] == "constant_or_numerically_unresolved"
            for report in reports.values()
        )
        spread[name] = {
            "replicate_means": [float(value) for value in means],
            "equal_weight_mean_of_replicate_means": float(np.mean(means)),
            "between_replicate_mean_variance": variance,
            "approximate_between_replicate_standard_error": (
                float(np.sqrt(variance / len(means)))
                if variance is not None and not unresolved
                else None
            ),
            "warnings": ["replicate_variance_unestimable"] if unresolved else [],
        }
    return {
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "replicate_count": len(reports),
        "replicates": reports,
        "between_replicate": spread,
        "replicate_independence": "caller_asserted_not_verified",
        "assumptions": list(ASSUMPTIONS),
        "equilibrium_qualification": "not_assessed",
        "physical_accuracy_validated": False,
        "hsp_predictions": None,
    }
