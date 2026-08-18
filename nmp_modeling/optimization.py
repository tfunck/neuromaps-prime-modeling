from dataclasses import dataclass

import numpy as np

from nmp_modeling.parametrization import FreeParam


@dataclass
class OptimizationResult:
    """Result from continuous optimization."""
    best_theta: dict
    best_loss: float
    history: np.ndarray
    n_iter: int
    n_evaluations: int
    method: str
    seed: int


def _parameter_space(free_params):
    """Return parameter names and normalized search-space values."""
    free_params = dict(free_params or {})
    if not free_params:
        raise ValueError("At least one free parameter is required.")

    for name, param in free_params.items():
        if not isinstance(param, FreeParam):
            raise TypeError(f"Free parameter '{name}' must be a FreeParam object.")

    names = list(free_params)
    lower = np.asarray([free_params[name].bounds[0] for name in names], dtype=float)
    upper = np.asarray([free_params[name].bounds[1] for name in names], dtype=float)
    initial = np.asarray([free_params[name].init for name in names], dtype=float)
    initial = (initial - lower) / (upper - lower)
    return names, lower, upper, initial


def _decode_theta(values, names, lower, upper):
    """Convert normalized coordinates to named physical parameter values."""
    values = lower + np.asarray(values, dtype=float) * (upper - lower)
    return {name: float(value) for name, value in zip(names, values)}


def _run_pso(evaluate, initial, seed, max_iter, population_size, options, verbose):
    """Run standard global-best PSO in normalized coordinates."""
    options = dict(options or {})
    inertia = float(options.pop("inertia", 0.729))
    cognitive = float(options.pop("cognitive", 1.49445))
    social = float(options.pop("social", 1.49445))
    if options:
        raise ValueError(f"Unknown PSO option(s): {sorted(options)}")

    rng = np.random.default_rng(seed)
    position = rng.random((population_size, initial.size))
    position[0] = initial
    velocity = rng.uniform(-0.1, 0.1, size=position.shape)

    values = np.asarray([evaluate(x) for x in position])
    personal_best = position.copy()
    personal_values = values.copy()
    best_idx = int(np.argmin(values))
    global_best = position[best_idx].copy()
    global_value = float(values[best_idx])
    history = [global_value]
    n_evaluations = population_size

    if verbose:
        print(f"1/{max_iter} loss={global_value:.6g}")

    for iteration in range(2, max_iter + 1):
        r1 = rng.random(position.shape)
        r2 = rng.random(position.shape)
        velocity = (
            inertia * velocity
            + cognitive * r1 * (personal_best - position)
            + social * r2 * (global_best - position)
        )

        proposed = position + velocity
        outside = (proposed < 0.0) | (proposed > 1.0)
        position = np.clip(proposed, 0.0, 1.0)
        velocity[outside] = 0.0

        values = np.asarray([evaluate(x) for x in position])
        n_evaluations += population_size
        improved = values < personal_values
        personal_best[improved] = position[improved]
        personal_values[improved] = values[improved]

        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()

        history.append(global_value)
        if verbose:
            print(f"{iteration}/{max_iter} loss={global_value:.6g}")

    return global_best, global_value, history, n_evaluations


def _run_cmaes(evaluate, initial, seed, max_iter, population_size, options, verbose):
    """Run CMA-ES in normalized coordinates."""
    try:
        from cmaes import CMA
    except ImportError as exc:
        raise ImportError("CMA-ES requires the 'cmaes' package.") from exc

    options = dict(options or {})
    sigma = float(options.pop("sigma", 0.2))
    if options:
        raise ValueError(f"Unknown CMA-ES option(s): {sorted(options)}")

    kwargs = {} if population_size is None else {"population_size": int(population_size)}
    optimizer = CMA(
        mean=np.clip(initial, 1e-12, 1.0 - 1e-12),
        sigma=sigma,
        bounds=np.tile([0.0, 1.0], (initial.size, 1)),
        seed=int(seed),
        **kwargs,
    )

    best_x = initial.copy()
    best_loss = np.inf
    history = []
    n_evaluations = 0

    for iteration in range(1, max_iter + 1):
        solutions = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            loss = evaluate(x)
            solutions.append((x, loss))
            n_evaluations += 1

            if loss < best_loss:
                best_x = np.asarray(x, dtype=float).copy()
                best_loss = float(loss)

        optimizer.tell(solutions)
        history.append(best_loss)

        if verbose:
            print(f"{iteration}/{max_iter} loss={best_loss:.6g}")

        if optimizer.should_stop():
            break

    return best_x, best_loss, history, n_evaluations


def optimize(
    objective,
    free_params,
    method="cmaes",
    seed=0,
    max_iter=200,
    population_size=None,
    method_options=None,
    verbose=False,
):
    """Minimize a scalar objective over bounded continuous parameters."""
    if not callable(objective):
        raise TypeError("objective must be callable.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive.")

    names, lower, upper, initial = _parameter_space(free_params)
    method = str(method).lower()

    if method not in {"pso", "cmaes"}:
        raise ValueError("method must be 'pso' or 'cmaes'.")

    if population_size is None and method == "pso":
        population_size = max(20, 4 * len(names))
    if population_size is not None:
        population_size = int(population_size)
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")

    def evaluate(values):
        loss = float(objective(_decode_theta(values, names, lower, upper)))
        if not np.isfinite(loss):
            return np.inf
        return loss

    if method == "pso":
        best_x, best_loss, history, n_evaluations = _run_pso(
            evaluate, initial, int(seed), int(max_iter), population_size, method_options, verbose
        )
    else:
        best_x, best_loss, history, n_evaluations = _run_cmaes(
            evaluate, initial, int(seed), int(max_iter), population_size, method_options, verbose
        )

    return OptimizationResult(
        best_theta=_decode_theta(best_x, names, lower, upper),
        best_loss=best_loss,
        history=np.asarray(history, dtype=float),
        n_iter=len(history),
        n_evaluations=n_evaluations,
        method=method,
        seed=int(seed),
    )
