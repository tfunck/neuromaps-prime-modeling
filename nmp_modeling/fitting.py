from dataclasses import dataclass
import itertools

import numpy as np


@dataclass
class EvaluationResult:
    """Result from evaluating one parameter setting."""
    theta: dict
    loss: float
    run_losses: np.ndarray
    target_losses: np.ndarray
    target_names: list
    target_weights: np.ndarray


@dataclass
class SweepResult:
    """Result from a grid sweep."""
    grid: dict
    losses: np.ndarray
    best_theta: dict
    best_loss: float
    run_losses: np.ndarray
    target_losses: np.ndarray
    fixed: dict
    target_names: list
    target_weights: np.ndarray

    @property
    def params(self):
        """Return fixed parameters merged with the best grid parameters."""
        return {**self.fixed, **self.best_theta}


def _as_list(x):
    """Return x as a non-empty list."""
    if isinstance(x, (list, tuple)):
        out = list(x)
    else:
        out = [x]

    if len(out) == 0:
        raise ValueError("At least one target is required.")

    return out


def _target_names(targets):
    """Return target names from EmpiricalTarget-like objects."""
    return [t.spec.name for t in targets]


def _target_weights(targets, target_weights=None):
    """Return target weights as a float array."""
    names = _target_names(targets)

    if target_weights is None:
        return np.ones(len(targets), dtype=float)

    if np.isscalar(target_weights):
        return np.repeat(float(target_weights), len(targets))

    if isinstance(target_weights, dict):
        unknown = set(target_weights) - set(names)
        if unknown:
            raise ValueError(f"Unknown target weight name(s): {sorted(unknown)}")
        return np.asarray([target_weights.get(name, 0.0) for name in names], dtype=float)

    weights = np.asarray(target_weights, dtype=float)

    if weights.shape != (len(targets),):
        raise ValueError("target_weights must have one value per target.")

    return weights


def _subject_seeds(run_seed, n_subjects):
    """Generate deterministic subject seeds from one run seed."""
    return [int(run_seed) * 100000 + i for i in range(n_subjects)]


def evaluate_theta(
    adapter,
    theta,
    targets,
    n_subjects=1,
    run_seeds=(0,),
    target_weights=None,
):
    """Evaluate one parameter setting across targets, seeds, and subjects."""
    targets = _as_list(targets)
    run_seeds = list(run_seeds)
    weights = _target_weights(targets, target_weights)
    names = _target_names(targets)

    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1.")
    if len(run_seeds) == 0:
        raise ValueError("At least one run seed is required.")

    run_losses = np.zeros(len(run_seeds), dtype=float)
    target_losses = np.zeros((len(run_seeds), len(targets)), dtype=float)

    for r, run_seed in enumerate(run_seeds):
        values_by_target = [[] for _ in targets]

        for subject_seed in _subject_seeds(run_seed, n_subjects):
            simulated = adapter.simulate(theta, subject_seed)

            for k, target in enumerate(targets):
                values_by_target[k].append(target.observable(simulated))

        for k, target in enumerate(targets):
            aggregated = target.aggregate_observable(values_by_target[k])
            target_losses[r, k] = target.distance(aggregated)

        run_losses[r] = float(np.sum(weights * target_losses[r]))

    return EvaluationResult(
        theta=dict(theta),
        loss=float(np.mean(run_losses)),
        run_losses=run_losses,
        target_losses=target_losses,
        target_names=names,
        target_weights=weights,
    )


def grid_sweep(
    adapter,
    free_grid,
    fixed,
    targets,
    n_subjects=1,
    run_seeds=(0,),
    target_weights=None,
    verbose=True,
):
    """Run a simple grid sweep using EmpiricalTarget objects."""
    targets = _as_list(targets)
    run_seeds = list(run_seeds)

    fixed = dict(fixed or {})
    grid = {k: np.asarray(v, dtype=float) for k, v in dict(free_grid or {}).items()}
    overlap = set(fixed) & set(grid)
    if overlap:
        raise ValueError(
            f"Parameter(s) cannot appear in both fixed and free_grid: {sorted(overlap)}"
        )

    for name, values in grid.items():
        if values.size == 0:
            raise ValueError(f"Grid for parameter '{name}' is empty.")

    names = list(grid)
    values = [grid[name] for name in names]
    shape = tuple(len(v) for v in values) if names else (1,)
    n_points = int(np.prod(shape))

    weights = _target_weights(targets, target_weights)
    target_names = _target_names(targets)

    losses = np.full(shape, np.nan)
    run_losses = np.full(shape + (len(run_seeds),), np.nan)
    target_losses = np.full(shape + (len(run_seeds), len(targets)), np.nan)

    combos = itertools.product(*values) if names else [()]

    for flat_idx, combo in enumerate(combos):
        idx = np.unravel_index(flat_idx, shape)
        theta = {**fixed, **dict(zip(names, combo))}

        result = evaluate_theta(
            adapter=adapter,
            theta=theta,
            targets=targets,
            n_subjects=n_subjects,
            run_seeds=run_seeds,
            target_weights=weights,
        )

        losses[idx] = result.loss
        run_losses[idx] = result.run_losses
        target_losses[idx] = result.target_losses

        if verbose:
            print(f"{flat_idx + 1}/{n_points} theta={theta} loss={result.loss:.6g}")

    best_idx = np.unravel_index(int(np.nanargmin(losses)), shape)
    best_theta = {name: float(values[i][best_idx[i]]) for i, name in enumerate(names)}

    return SweepResult(
        grid=grid,
        losses=losses,
        best_theta=best_theta,
        best_loss=float(losses[best_idx]),
        run_losses=run_losses,
        target_losses=target_losses,
        fixed=fixed,
        target_names=target_names,
        target_weights=weights,
    )
