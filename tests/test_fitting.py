"""Tests for the fitting layer (grid_sweep, SweepResult).

These tests use a mock adapter so they don't require Neuronumba. The mock
adapter simulates synthetic BOLD whose statistics depend on theta in a
controlled way, which lets us verify the sweep mechanics.
"""

import numpy as np
import pytest

from nmp_modeling.fitting import grid_sweep
from nmp_modeling import observables


class MockAdapter:
    """Synthetic adapter for testing.

    Returns a deterministic BOLD-shaped array whose FC matrix has a known,
    monotonic relationship to the parameter distance from a 'truth' point.
    This lets us write reliable tests without depending on stochastic mock
    signals being noisy enough.

    The trick: we precompute a 'truth' FC matrix, and at each call we
    return BOLD that *exactly* induces a target FC equal to a convex
    combination of the truth FC and an unrelated reference FC. The
    combination weight is a deterministic function of theta's distance
    from the truth point. So the closer theta is to truth, the more the
    induced FC matches the truth FC, with no stochastic flakiness.
    """

    def __init__(self, n_nodes=20, n_timepoints=400, true_a=0.3, true_b=0.7):
        self.n_nodes = n_nodes
        self.n_timepoints = n_timepoints
        self.true_a = true_a
        self.true_b = true_b
        self.call_count = 0

        # Precompute two reference signals: one that defines the "truth"
        # correlation structure, one that defines an unrelated one.
        rng_truth = np.random.default_rng(12345)
        self._truth_signal = rng_truth.standard_normal((n_timepoints, n_nodes))
        rng_other = np.random.default_rng(54321)
        self._other_signal = rng_other.standard_normal((n_timepoints, n_nodes))

    def simulate(self, theta, seed):
        self.call_count += 1
        a = theta.get("a", 0.0)
        b = theta.get("b", 0.0)
        # Distance-from-truth in [0, 1] via squared error scaled.
        err = (a - self.true_a) ** 2 + (b - self.true_b) ** 2
        # Saturating function: when theta == truth, weight = 0 (pure truth).
        # When theta is far, weight -> 1 (pure 'other'). Deterministic.
        weight = err / (err + 0.05)   # err=0 -> 0, err=0.5 -> ~0.91
        # Linear blend of two deterministic signals. Tiny seed-dependent
        # jitter for realism, but small enough that loss is dominated by
        # the deterministic blend.
        rng = np.random.default_rng(seed)
        jitter = rng.standard_normal((self.n_timepoints, self.n_nodes)) * 0.01
        return (1 - weight) * self._truth_signal + weight * self._other_signal + jitter


def test_grid_sweep_returns_correct_shapes():
    """SweepResult arrays have the documented shapes."""
    adapter = MockAdapter()
    empirical = adapter.simulate({"a": 0.3, "b": 0.7, "G": 1.0}, seed=0)
    emp_fc = observables.fc_fisher_z(empirical)

    result = grid_sweep(
        adapter=adapter,
        free_grid={
            "a": np.linspace(0, 0.6, 4),
            "b": np.linspace(0, 1.0, 5),
        },
        fixed={"G": 1.0},
        observable=observables.fc_fisher_z,
        empirical_target=emp_fc,
        distance=observables.pearson_lower_triangle,
        n_subjects=3,
        run_seeds=[1, 2, 3],
        aggregate_observable=lambda obs_list: np.mean(obs_list, axis=0),
        verbose=False,
    )

    assert result.losses.shape == (4, 5)
    assert result.all_seed_losses.shape == (4, 5, 3)
    assert set(result.best_theta.keys()) == {"a", "b"}
    assert result.fixed == {"G": 1.0}


def test_grid_sweep_recovers_known_optimum():
    """A clearly noise-dominated mock landscape: sweep should pick the
    point closest to the truth."""
    adapter = MockAdapter(true_a=0.3, true_b=0.7)
    empirical = adapter.simulate({"a": 0.3, "b": 0.7, "G": 1.0}, seed=999)
    emp_fc = observables.fc_fisher_z(empirical)

    grid_a = np.array([0.0, 0.3, 0.6])
    grid_b = np.array([0.0, 0.4, 0.7, 1.0])

    result = grid_sweep(
        adapter=adapter,
        free_grid={"a": grid_a, "b": grid_b},
        fixed={"G": 1.0},
        observable=observables.fc_fisher_z,
        empirical_target=emp_fc,
        distance=observables.pearson_lower_triangle,
        n_subjects=10,
        run_seeds=[10, 20, 30],
        aggregate_observable=lambda obs_list: np.mean(obs_list, axis=0),
        verbose=False,
    )

    # The grid points 0.3 (a) and 0.7 (b) match the truth exactly,
    # so the mock landscape should have its minimum there.
    assert result.best_theta["a"] == 0.3
    assert result.best_theta["b"] == 0.7


def test_params_property_merges_fixed_and_best():
    """SweepResult.params merges fixed + best_theta — used for stage threading."""
    adapter = MockAdapter()
    empirical = adapter.simulate({"a": 0.3, "b": 0.7, "G": 1.0}, seed=0)
    emp_fc = observables.fc_fisher_z(empirical)

    result = grid_sweep(
        adapter=adapter,
        free_grid={"a": [0.0, 0.3]},
        fixed={"G": 1.5, "b": 0.5},
        observable=observables.fc_fisher_z,
        empirical_target=emp_fc,
        distance=observables.pearson_lower_triangle,
        n_subjects=2,
        run_seeds=[1],
        aggregate_observable=lambda obs_list: np.mean(obs_list, axis=0),
        verbose=False,
    )

    params = result.params
    assert "G" in params
    assert "b" in params
    assert "a" in params
    assert params["G"] == 1.5
    assert params["b"] == 0.5


def test_grid_sweep_calls_simulate_correct_number_of_times():
    """Verify the call structure: n_grid_points * n_seeds * n_subjects."""
    adapter = MockAdapter()
    empirical = adapter.simulate({"a": 0.3, "b": 0.7, "G": 1.0}, seed=0)
    emp_fc = observables.fc_fisher_z(empirical)

    n_a, n_b = 3, 4
    n_subjects = 5
    n_seeds = 2
    expected_extra = n_a * n_b * n_seeds * n_subjects

    pre_count = adapter.call_count
    grid_sweep(
        adapter=adapter,
        free_grid={"a": np.linspace(0, 0.6, n_a), "b": np.linspace(0, 1.0, n_b)},
        fixed={"G": 1.0},
        observable=observables.fc_fisher_z,
        empirical_target=emp_fc,
        distance=observables.pearson_lower_triangle,
        n_subjects=n_subjects,
        run_seeds=list(range(n_seeds)),
        aggregate_observable=lambda obs_list: np.mean(obs_list, axis=0),
        verbose=False,
    )
    actual_extra = adapter.call_count - pre_count
    assert actual_extra == expected_extra


def test_stage_threading():
    """Two-stage workflow: stage-2 fixed includes stage-1's best."""
    adapter = MockAdapter(true_a=0.3, true_b=0.7)
    empirical = adapter.simulate({"a": 0.3, "b": 0.7, "G": 1.0}, seed=0)
    emp_fc = observables.fc_fisher_z(empirical)

    # Stage 1: fit a, hold b at 0.
    stage1 = grid_sweep(
        adapter=adapter,
        free_grid={"a": [0.0, 0.3, 0.6]},
        fixed={"G": 1.0, "b": 0.0},
        observable=observables.fc_fisher_z,
        empirical_target=emp_fc,
        distance=observables.pearson_lower_triangle,
        n_subjects=5,
        run_seeds=[1, 2],
        aggregate_observable=lambda obs_list: np.mean(obs_list, axis=0),
        verbose=False,
    )

    # Stage 2: fit b, hold a at stage-1's best.
    stage2_fixed = {k: v for k, v in stage1.params.items() if k != "b"}
    stage2 = grid_sweep(
        adapter=adapter,
        free_grid={"b": [0.0, 0.4, 0.7]},
        fixed=stage2_fixed,
        observable=observables.fc_fisher_z,
        empirical_target=emp_fc,
        distance=observables.pearson_lower_triangle,
        n_subjects=5,
        run_seeds=[3, 4],
        aggregate_observable=lambda obs_list: np.mean(obs_list, axis=0),
        verbose=False,
    )

    # Stage 2's fixed dict carries forward stage 1's a value.
    assert "a" in stage2.fixed
    assert stage2.fixed["a"] == stage1.best_theta["a"]