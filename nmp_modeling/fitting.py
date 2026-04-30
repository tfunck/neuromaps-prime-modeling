"""Layer 3b: backend-agnostic fitting primitives.

This module knows nothing about Neuronumba (or any other backend). It
treats ``adapter.simulate(theta, seed) -> bold`` as a black box, which
makes the fitting machinery reusable across backends.

V1 ships with grid sweep only. Future versions can add scipy.optimize
wrappers, CMA-ES, differential evolution, etc. — they would live alongside
``grid_sweep`` with the same ``(adapter, observable, empirical_target,
distance)`` signature.

Note: Zhang et al. 2024 (PNAS) fit 10 parameters of a heterogeneous FIC
model using CMA-ES. Grid sweep would be intractable for that case; CMA-ES
is the obvious next addition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import itertools
import numpy as np


@dataclass
class SweepResult:
    """Result of a grid sweep over one or more free parameters.

    Attributes
    ----------
    grid : dict[str, np.ndarray]
        The grid that was swept, by parameter name.
    losses : np.ndarray
        Loss values, shape ``tuple(len(grid[name]) for name in grid)``.
        Axes correspond to grid keys in their dict iteration order.
    best_theta : dict[str, float]
        Combination of ``grid`` values minimising mean loss across seeds.
        Suitable for passing as ``fixed`` into a subsequent stage.
    best_loss : float
        Loss at ``best_theta``.
    all_seed_losses : np.ndarray
        Per-seed losses, shape ``losses.shape + (n_seeds,)``. Useful for
        diagnosing seed-to-seed variance.
    fixed : dict[str, float]
        Fixed parameters used for this sweep (carried through for record).
    """
    grid: Dict[str, np.ndarray]
    losses: np.ndarray
    best_theta: Dict[str, float]
    best_loss: float
    all_seed_losses: np.ndarray
    fixed: Dict[str, float]

    @property
    def params(self) -> Dict[str, float]:
        """All resolved parameter values (best_theta + fixed). Convenient
        to splat into the next stage's ``fixed`` argument."""
        return {**self.fixed, **self.best_theta}


def grid_sweep(
    adapter,
    free_grid: Dict[str, Sequence[float]],
    fixed: Dict[str, float],
    observable: Callable[[np.ndarray], np.ndarray],
    empirical_target: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    n_subjects: int,
    run_seeds: Sequence[int],
    aggregate_observable: Optional[Callable] = None,
    verbose: bool = True,
) -> SweepResult:
    """Run a grid sweep over free parameters and return the minimum-loss point.

    The fitting protocol — multi-seed averaging, multi-subject simulation,
    aggregation of observables across subjects — is intentionally
    opinionated and matches the pattern used in the original Neuronumba
    notebook (and Deco-style modelling more generally)::

        For each grid point theta:
            For each run_seed (separate stochastic runs to denoise):
                For each of n_subjects synthetic subjects (seed = run_seed*1000+i):
                    bold = adapter.simulate(theta, subject_seed)
                    obs[subject] = observable(bold)
                aggregated_obs = aggregate_observable(obs[0..n_subjects-1])
                loss[seed] = distance(aggregated_obs, empirical_target)
            mean_loss[theta] = mean(loss across seeds)
        best_theta = argmin(mean_loss)

    Parameters
    ----------
    adapter : object with .simulate(theta, seed) -> bold
        Any backend adapter. Need not be a NeuronumbaAdapter.
    free_grid : dict[str, sequence of float]
        Each entry gives the values to test for one free parameter. The
        Cartesian product is swept. For a single fixed value, pass a
        one-element list (e.g. ``{"a": [0.0]}``).
    fixed : dict[str, float]
        Parameters held constant across the entire sweep. Merged with each
        grid point before being passed to the adapter.
    observable : callable
        ``observable(bold) -> array``. Computed on each simulated subject's
        BOLD.
    empirical_target : np.ndarray
        Precomputed observable on empirical data. Compared against the
        aggregated simulated observable via ``distance``.
    distance : callable
        ``distance(simulated, empirical) -> float``. Smaller = better fit.
    n_subjects : int
        Number of synthetic subjects to simulate per (theta, run_seed)
        combination.
    run_seeds : sequence of int
        Outer seeds; each one defines an independent group of synthetic
        subjects (the i-th subject seed within run_seed s is s*1000 + i).
        Multiple run_seeds let you average out stochastic noise in the
        loss landscape.
    aggregate_observable : callable, optional
        How to combine per-subject observables into a single object suitable
        for ``distance``. Default: numpy concatenation along axis 0, which
        is correct for FCD distributions. For FC matrices, you'd want
        ``lambda obs_list: np.mean(obs_list, axis=0)`` instead.
    verbose : bool
        Print progress per grid point.

    Returns
    -------
    SweepResult
    """
    if aggregate_observable is None:
        # Default: concatenate. Correct for distribution-style observables
        # like FCD (which the KS distance compares as samples). For FC
        # matrices you'd pass an explicit mean.
        aggregate_observable = lambda obs_list: np.concatenate(
            [np.ravel(o) for o in obs_list]
        )

    # Lock down a deterministic grid axis order for the output array.
    grid_names = list(free_grid.keys())
    grid_values = [np.asarray(free_grid[n], dtype=float) for n in grid_names]
    grid_shape = tuple(len(v) for v in grid_values)

    n_seeds = len(run_seeds)

    losses = np.full(grid_shape, np.nan)
    all_seed_losses = np.full(grid_shape + (n_seeds,), np.nan)

    # Iterate over the Cartesian product of grid points.
    # itertools.product preserves the order of grid_names.
    for flat_idx, combo in enumerate(itertools.product(*grid_values)):
        # Multi-dim index into the output arrays.
        idx = np.unravel_index(flat_idx, grid_shape)

        # Build theta for this grid point.
        theta_grid = dict(zip(grid_names, combo))
        theta = {**fixed, **theta_grid}

        if verbose:
            theta_str = ", ".join(f"{k}={v:.4g}" for k, v in theta_grid.items())
            print(f"[grid_sweep] {theta_str}")

        # Run the seed loop.
        seed_losses = []
        for s_i, run_seed in enumerate(run_seeds):
            # Per-subject seeds: a deterministic but well-spaced family.
            subj_seeds = [int(run_seed) * 1000 + i for i in range(n_subjects)]

            obs_list = []
            for subj_seed in subj_seeds:
                bold = adapter.simulate(theta, subj_seed)
                obs_list.append(observable(bold))

            sim_aggregate = aggregate_observable(obs_list)
            seed_loss = distance(sim_aggregate, empirical_target)
            seed_losses.append(seed_loss)
            all_seed_losses[idx + (s_i,)] = seed_loss

        mean_loss = float(np.mean(seed_losses))
        losses[idx] = mean_loss

        if verbose:
            print(f"            mean_loss={mean_loss:.6f}")

    # Locate the minimum.
    best_flat = int(np.argmin(losses))
    best_idx = np.unravel_index(best_flat, grid_shape)
    best_theta = {n: float(grid_values[i][best_idx[i]]) for i, n in enumerate(grid_names)}

    return SweepResult(
        grid={n: grid_values[i] for i, n in enumerate(grid_names)},
        losses=losses,
        best_theta=best_theta,
        best_loss=float(losses[best_idx]),
        all_seed_losses=all_seed_losses,
        fixed=dict(fixed),
    )