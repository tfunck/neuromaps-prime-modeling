import numpy as np
from dataclasses import dataclass

from nmp_modeling.hopf.preprocessing import ensure_subject_time_node
from nmp_modeling.observables import (
    check_time_series,
    compute_fc,
    compute_normalized_shifted_covariance,
    compute_forward_reverse_mi,
    gaussian_mi_transform,
)


@dataclass
class LagcovObservables:
    """Container for FC and normalized shifted covariance."""
    fc: np.ndarray
    normalized_shifted_covariance: np.ndarray


@dataclass
class SimulatedLagcovEvaluation:
    """Container for simulation-based lagcov evaluation."""
    observables: LagcovObservables
    n_runs: int


@dataclass
class MIObservables:
    """Container for MI-transformed FC and time-reversal observables."""
    fc_mi: np.ndarray
    forward_mi: np.ndarray
    reverse_mi: np.ndarray


@dataclass
class SimulatedMIEvaluation:
    """Container for simulation-based MI observable evaluation."""
    observables: MIObservables
    n_runs: int


def _seed_list(seeds):
    """Validate and return a list of integer seeds."""
    seeds = list(seeds)
    if len(seeds) == 0:
        raise ValueError("At least one seed is required.")
    return [int(seed) for seed in seeds]


def _simulate_runs(adapter_factory, weights, theta, seeds):
    """Run an adapter over multiple random seeds."""
    if adapter_factory is None:
        raise ValueError("adapter_factory must be provided.")
    adapter = adapter_factory(weights)
    return [adapter.simulate(theta, seed=seed) for seed in _seed_list(seeds)]


def average_lagcov_observables(timeseries, lag=1, preprocess_fn=None):
    """Compute average FC and normalized shifted covariance across runs."""
    arr, _ = ensure_subject_time_node(timeseries)
    fcs = []
    shifted_covs = []

    for s in range(arr.shape[0]):
        ts = arr[s]
        if preprocess_fn is not None:
            ts = preprocess_fn(ts)
        ts = check_time_series(ts)
        fcs.append(compute_fc(ts))
        shifted_covs.append(compute_normalized_shifted_covariance(ts, lag=lag))

    return LagcovObservables(
        fc=np.mean(np.asarray(fcs, dtype=float), axis=0),
        normalized_shifted_covariance=np.mean(
            np.asarray(shifted_covs, dtype=float),
            axis=0,
        ),
    )


def simulate_lagcov_observables(
    adapter_factory,
    weights,
    theta,
    seeds,
    lag=1,
    preprocess_fn=None,
):
    """Simulate runs and compute average lagcov observables."""
    simulated = _simulate_runs(
        adapter_factory=adapter_factory,
        weights=weights,
        theta=theta,
        seeds=seeds,
    )

    observables = average_lagcov_observables(
        simulated,
        lag=lag,
        preprocess_fn=preprocess_fn,
    )

    return SimulatedLagcovEvaluation(
        observables=observables,
        n_runs=len(seeds),
    )


def _fc_mi_from_timeseries(timeseries, eps=1e-12):
    """Compute Gaussian-MI-transformed zero-lag FC."""
    fc = compute_fc(timeseries)
    fc_mi = gaussian_mi_transform(fc, eps=eps)
    np.fill_diagonal(fc_mi, 0.0)
    return fc_mi


def _forward_reverse_mi_from_timeseries(timeseries, lag=2, eps=1e-12):
    """Compute Gaussian-MI-transformed forward and reverse lagged dependencies."""
    forward_mi, reverse_mi, _ = compute_forward_reverse_mi(
        timeseries,
        lag=lag,
        eps=eps,
    )
    np.fill_diagonal(forward_mi, 0.0)
    np.fill_diagonal(reverse_mi, 0.0)
    return forward_mi, reverse_mi


def average_mi_observables(timeseries, lag=2, preprocess_fn=None, eps=1e-12):
    """Compute average MI observables across runs."""
    arr, _ = ensure_subject_time_node(timeseries)
    fc_mis = []
    forward_mis = []
    reverse_mis = []

    for s in range(arr.shape[0]):
        ts = arr[s]
        if preprocess_fn is not None:
            ts = preprocess_fn(ts)
        ts = check_time_series(ts)
        fc_mis.append(_fc_mi_from_timeseries(ts, eps=eps))
        forward_mi, reverse_mi = _forward_reverse_mi_from_timeseries(
            ts,
            lag=lag,
            eps=eps,
        )
        forward_mis.append(forward_mi)
        reverse_mis.append(reverse_mi)

    return MIObservables(
        fc_mi=np.mean(np.asarray(fc_mis, dtype=float), axis=0),
        forward_mi=np.mean(np.asarray(forward_mis, dtype=float), axis=0),
        reverse_mi=np.mean(np.asarray(reverse_mis, dtype=float), axis=0),
    )


def simulate_mi_observables(
    adapter_factory,
    weights,
    theta,
    seeds,
    lag=2,
    preprocess_fn=None,
    eps=1e-12,
):
    """Simulate runs and compute average MI observables."""
    simulated = _simulate_runs(
        adapter_factory=adapter_factory,
        weights=weights,
        theta=theta,
        seeds=seeds,
    )

    observables = average_mi_observables(
        simulated,
        lag=lag,
        preprocess_fn=preprocess_fn,
        eps=eps,
    )

    return SimulatedMIEvaluation(
        observables=observables,
        n_runs=len(seeds),
    )
