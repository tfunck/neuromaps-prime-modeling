import numpy as np
from dataclasses import dataclass

from nmp_modeling.hopf.preprocessing import ensure_subject_time_node
from nmp_modeling.observables import (
    check_time_series,
    compute_fc,
    compute_normalized_shifted_covariance,
)


@dataclass
class LagcovObservables:
    """Container for FC and normalized shifted covariance."""
    fc: np.ndarray
    normalized_shifted_covariance: np.ndarray


@dataclass
class SimulatedLagcovEvaluation:
    """Container for simulation-based Hopf lagcov evaluation."""
    observables: LagcovObservables
    n_runs: int


def average_lagcov_observables(timeseries, lag=1, preprocess_fn=None):
    """Compute average FC and normalized shifted covariance across subjects/runs."""
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
    """Simulate nonlinear Hopf runs and compute average lagcov observables.

    adapter_factory must be a callable:
        adapter = adapter_factory(weights)

    The returned adapter must expose:
        adapter.simulate(theta, seed)
    """
    if adapter_factory is None:
        raise ValueError("adapter_factory must be provided.")

    seeds = list(seeds)
    if len(seeds) == 0:
        raise ValueError("At least one seed is required.")

    adapter = adapter_factory(weights)

    fcs = []
    shifted_covs = []

    for seed in seeds:
        ts = adapter.simulate(theta, seed=int(seed))
        if preprocess_fn is not None:
            ts = preprocess_fn(ts)
        ts = check_time_series(ts)
        fcs.append(compute_fc(ts))
        shifted_covs.append(compute_normalized_shifted_covariance(ts, lag=lag))

    observables = LagcovObservables(
        fc=np.mean(np.asarray(fcs, dtype=float), axis=0),
        normalized_shifted_covariance=np.mean(
            np.asarray(shifted_covs, dtype=float),
            axis=0,
        ),
    )

    return SimulatedLagcovEvaluation(
        observables=observables,
        n_runs=len(seeds),
    )
