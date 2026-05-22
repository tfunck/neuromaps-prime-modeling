import numpy as np
from scipy.optimize import brentq

from neuronumba.tools.random import set_seed
from neuronumba.simulator.simulator import simulate_nodelay


_DECO_CURRENT_OFFSET = -0.026


def _as_region_vector(value, n_nodes, name):
    """Return a scalar or vector value as a region-level vector."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_nodes, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != n_nodes:
        raise ValueError(f"{name} must be a scalar or have length {n_nodes}.")
    return arr.astype(float, copy=False)


def _effective_exc_gain(model, n_nodes):
    """Compute region-specific excitatory gain for GenericBEIDMF-like models."""
    base_gain = _as_region_vector(getattr(model, "M_e", 1.0), n_nodes, "M_e")
    gain_e = _as_region_vector(getattr(model, "gain_e", 0.0), n_nodes, "gain_e")
    gain_map = _as_region_vector(
        getattr(model, "gain_map_e", np.zeros(n_nodes)),
        n_nodes,
        "gain_map_e",
    )
    eff_gain = base_gain * (1.0 + gain_e * gain_map)
    if np.any(eff_gain <= 0):
        raise ValueError("Effective excitatory gain must be positive.")
    return eff_gain, base_gain


def _rate_from_argument(x, de):
    """Evaluate x / (1 - exp(-de * x)) with the correct zero limit."""
    if abs(x) < 1e-12:
        return 1.0 / de
    return x / (1.0 - np.exp(-de * x))


def _solve_argument_for_rate(rate, de):
    """Solve x / (1 - exp(-de * x)) = rate."""
    rate = float(rate)
    if rate <= 0:
        raise ValueError("target_rate values must be positive.")

    zero_limit = 1.0 / de
    if np.isclose(rate, zero_limit):
        return 0.0

    def objective(x):
        return _rate_from_argument(x, de) - rate

    low = -1.0
    high = max(1.0, 2.0 * rate)
    while objective(low) > 0:
        low *= 2.0
        if low < -1e6:
            raise RuntimeError("Failed to bracket the transfer-function root.")
    while objective(high) < 0:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("Failed to bracket the transfer-function root.")

    return float(brentq(objective, low, high))


def _target_current_offset(model, n_nodes, target_rate=None):
    """Compute the target for Ie - be / ae."""
    eff_gain, base_gain = _effective_exc_gain(model, n_nodes)

    if target_rate is None:
        return _DECO_CURRENT_OFFSET * base_gain / eff_gain

    rates = _as_region_vector(target_rate, n_nodes, "target_rate")
    arguments = np.array(
        [_solve_argument_for_rate(rate, float(model.de)) for rate in rates],
        dtype=float,
    )

    return arguments / (float(model.ae) * eff_gain)


def compute_j(
    model,
    weights,
    g,
    integrator,
    seed=None,
    target_rate=None,
    obs_var="Ie",
    t_max=10000.0,
    t_warmup=0.0,
    max_trials=5000,
    tolerance=0.005,
    min_step=0.005,
    verbose=False,
):
    """Compute Deco-style feedback inhibition J for a configured model."""
    set_seed(int(seed))

    sc = np.asarray(weights, dtype=float)

    if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
        raise ValueError("weights must be a square 2D matrix.")

    n_nodes = sc.shape[0]
    t_end = int(round(t_max))
    t_start = 1000 if t_end > 1000 else int(t_end / 10)

    if t_end <= t_start:
        raise ValueError("t_max is too short for FIC estimation.")

    model.configure(weights=sc, g=float(g))

    target_offset = _target_current_offset(
        model,
        n_nodes=n_nodes,
        target_rate=target_rate,
    )

    curr_j = np.ones(n_nodes, dtype=float)
    best_j = curr_j.copy()
    best_solved = -1
    best_largest_error = np.inf
    min_largest_error = np.inf
    slow_factor = 1.0

    for trial in range(max_trials):
        model.configure(J=curr_j)

        signal = simulate_nodelay(
            model,
            integrator,
            sc,
            obs_var,
            1.0,
            t_max,
            t_warmup,
        )

        current_offset = signal - (float(model.be) / float(model.ae))
        mean_offset = np.mean(current_offset[t_start:t_end, :], axis=0)
        error = mean_offset - target_offset
        abs_error = np.abs(error)
        largest_error = float(np.max(abs_error))
        solved = abs_error <= tolerance
        solved_count = int(np.sum(solved))

        if (
            solved_count > best_solved
            or solved_count == best_solved
            and largest_error < best_largest_error
        ):
            best_j = curr_j.copy()
            best_solved = solved_count
            best_largest_error = largest_error

        if verbose:
            print(
                f"trial={trial} solved={solved_count}/{n_nodes} "
                f"largest_error={largest_error:.6g}"
            )

        if solved_count == n_nodes:
            break

        if largest_error < min_largest_error:
            min_largest_error = largest_error
        else:
            slow_factor *= 0.5

        step = slow_factor * abs_error / 0.1
        step = np.where(abs_error > tolerance, np.maximum(step, min_step), 0.0)
        curr_j = curr_j + np.sign(error) * step

    return best_j
