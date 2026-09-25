"""Tests for the fitting layer (grid_sweep, SweepResult).

These tests use a mock adapter so they don't require Neuronumba. The mock
adapter returns a deterministic BOLD-shaped array whose FC matrix has a known,
monotonic relationship to the parameter distance from a 'truth' point.
"""

import numpy as np
import pytest

from nmp_modeling.fitting import grid_sweep
from nmp_modeling.observable_specs import make_empirical_target
from nmp_modeling import observables


class MockAdapter:
    def __init__(self, n_nodes=20, n_timepoints=400, true_a=0.3, true_b=0.7):
        self.n_nodes = n_nodes
        self.n_timepoints = n_timepoints
        self.true_a = true_a
        self.true_b = true_b
        self.call_count = 0
        rng_truth = np.random.default_rng(12345)
        self._truth_signal = rng_truth.standard_normal((n_timepoints, n_nodes))
        rng_other = np.random.default_rng(54321)
        self._other_signal = rng_other.standard_normal((n_timepoints, n_nodes))

    def simulate(self, theta, seed):
        self.call_count += 1
        a = theta.get("a", 0.0)
        b = theta.get("b", 0.0)
        err = (a - self.true_a) ** 2 + (b - self.true_b) ** 2
        weight = err / (err + 0.05)
        rng = np.random.default_rng(seed)
        jitter = rng.standard_normal((self.n_timepoints, self.n_nodes)) * 0.01
        return (1 - weight) * self._truth_signal + weight * self._other_signal + jitter


def _truth_target(adapter):
    """Empirical target = FC of the adapter's truth signal."""
    emp_fc = observables.compute_fc(adapter._truth_signal)
    return make_empirical_target(emp_fc, "fc_corr", input_type="observable")


def test_grid_sweep_returns_correct_shapes():
    adapter = MockAdapter()
    result = grid_sweep(
        adapter=adapter,
        free_grid={"a": np.linspace(0, 1, 4), "b": np.linspace(0, 1, 5)},
        fixed={"G": 1.0},
        targets=_truth_target(adapter),
        n_subjects=2,
        run_seeds=[1, 2, 3],
        verbose=False,
    )
    assert result.losses.shape == (4, 5)
    assert result.run_losses.shape == (4, 5, 3)
    assert set(result.best_theta.keys()) == {"a", "b"}
    assert result.fixed == {"G": 1.0}


def test_grid_sweep_recovers_known_optimum():
    adapter = MockAdapter(true_a=0.3, true_b=0.7)
    result = grid_sweep(
        adapter=adapter,
        free_grid={"a": [0.0, 0.3, 0.6], "b": [0.4, 0.7, 1.0]},
        fixed={"G": 1.0},
        targets=_truth_target(adapter),
        n_subjects=2,
        run_seeds=[1, 2],
        verbose=False,
    )
    assert result.best_theta["a"] == 0.3
    assert result.best_theta["b"] == 0.7


def test_params_property_merges_fixed_and_best():
    adapter = MockAdapter()
    result = grid_sweep(
        adapter=adapter,
        free_grid={"a": [0.1, 0.3]},
        fixed={"G": 1.5, "b": 0.5},
        targets=_truth_target(adapter),
        n_subjects=1,
        run_seeds=[7],
        verbose=False,
    )
    params = result.params
    assert params["G"] == 1.5 and params["b"] == 0.5 and "a" in params


def test_grid_sweep_calls_simulate_correct_number_of_times():
    adapter = MockAdapter()
    grid_sweep(
        adapter=adapter,
        free_grid={"a": [0.0, 0.5], "b": [0.0, 0.5, 1.0]},
        fixed={"G": 1.0},
        targets=_truth_target(adapter),
        n_subjects=3,
        run_seeds=[1, 2],
        verbose=False,
    )
    assert adapter.call_count == 2 * 3 * 3 * 2  # grid points x subjects x seeds


def test_fixed_and_grid_overlap_raises():
    adapter = MockAdapter()
    with pytest.raises(ValueError):
        grid_sweep(
            adapter=adapter,
            free_grid={"a": [0.0, 0.5]},
            fixed={"a": 1.0},
            targets=_truth_target(adapter),
            n_subjects=1,
            run_seeds=[1],
            verbose=False,
        )
