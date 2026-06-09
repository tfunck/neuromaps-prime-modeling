import numpy as np
from dataclasses import dataclass

from nmp_modeling.hopf.linear import evaluate_linear_hopf
from nmp_modeling.hopf.simulation import (
    LagcovObservables,
    average_lagcov_observables,
    simulate_lagcov_observables,
)


@dataclass
class LagcovGECResult:
    """Container for lagcov-style GEC fitting results."""
    gec: np.ndarray
    empirical_fc: np.ndarray
    empirical_normalized_shifted_covariance: np.ndarray
    simulated_fc: np.ndarray
    simulated_normalized_shifted_covariance: np.ndarray
    loss_history: list
    max_real_history: list
    n_iter: int
    converged: bool
    stop_reason: str


def _check_square_matrix(matrix, name):
    """Validate a square matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return arr


def _offdiag_values(matrix):
    """Return off-diagonal values from a square matrix."""
    mat = _check_square_matrix(matrix, "matrix")
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


def _offdiag_mse(a, b):
    """Return off-diagonal mean squared error between two square matrices."""
    return float(np.mean((_offdiag_values(a) - _offdiag_values(b)) ** 2))


def _make_update_mask(sc, update_mask=None, include_homologue_edges=False):
    """Create the binary edge-update mask."""
    sc = _check_square_matrix(sc, "sc")
    n_nodes = sc.shape[0]

    if update_mask is None:
        mask = sc > 0
    else:
        mask = np.asarray(update_mask, dtype=bool)
        if mask.shape != sc.shape:
            raise ValueError("update_mask must have the same shape as sc.")

    if include_homologue_edges:
        if n_nodes % 2 != 0:
            raise ValueError(
                "include_homologue_edges=True requires an even number of nodes."
            )
        offset = n_nodes // 2
        idx = np.arange(n_nodes)
        mask[idx, (idx + offset) % n_nodes] = True

    np.fill_diagonal(mask, False)
    return mask


def _initialize_gec(sc, update_mask, init="scaled_sc", initial_gec=None, max_c=0.2):
    """Initialize the GEC matrix."""
    sc = _check_square_matrix(sc, "sc")

    if initial_gec is not None:
        gec = _check_square_matrix(initial_gec, "initial_gec").copy()
        if gec.shape != sc.shape:
            raise ValueError("initial_gec must have the same shape as sc.")
        gec[~update_mask] = 0.0
        np.fill_diagonal(gec, 0.0)
        return gec

    if init == "zeros":
        return np.zeros_like(sc, dtype=float)
    if init != "scaled_sc":
        raise ValueError("init must be 'scaled_sc' or 'zeros'.")

    positive_max = np.max(sc[update_mask]) if np.any(update_mask) else 0.0
    if positive_max <= 0:
        raise ValueError("Cannot initialize from scaled_sc with an empty/zero mask.")
    gec = sc / positive_max * float(max_c)
    gec[~update_mask] = 0.0
    np.fill_diagonal(gec, 0.0)
    return gec


def _normalize_to_max_c(gec, update_mask, max_c):
    """Normalize positive GEC values so their maximum equals max_c."""
    out = np.array(gec, dtype=float, copy=True)
    positive = out[update_mask]
    positive_max = np.max(positive) if positive.size else 0.0
    if positive_max > 0:
        out = out / positive_max * float(max_c)
    out[~update_mask] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _apply_l1_penalty(gec, update_mask, l1_alpha):
    """Apply Luppi-style sign-based L1 shrinkage on updated edges."""
    if l1_alpha <= 0:
        return gec
    out = np.array(gec, dtype=float, copy=True)
    out[update_mask] = out[update_mask] - float(l1_alpha) * np.sign(out[update_mask])
    return out


def _apply_constraints(
    gec,
    update_mask,
    allow_negative=False,
    l1_alpha=0.0,
    normalize_max=True,
    max_c=0.2,
):
    """Apply mask, optional L1 shrinkage, nonnegativity, and max normalization."""
    out = np.array(gec, dtype=float, copy=True)
    out[~update_mask] = 0.0
    np.fill_diagonal(out, 0.0)

    out = _apply_l1_penalty(
        gec=out,
        update_mask=update_mask,
        l1_alpha=float(l1_alpha),
    )
    if not allow_negative:
        out[out < 0.0] = 0.0
    if normalize_max and float(l1_alpha) == 0.0:
        out = _normalize_to_max_c(
            gec=out,
            update_mask=update_mask,
            max_c=max_c,
        )

    return out


def _prepare_empirical_observables(
    empirical_timeseries=None,
    empirical_fc=None,
    empirical_normalized_shifted_covariance=None,
    lag=1,
    preprocess_fn=None,
):
    """Create empirical FC and normalized shifted covariance targets."""
    if empirical_fc is not None and empirical_normalized_shifted_covariance is not None:
        fc = _check_square_matrix(empirical_fc, "empirical_fc")
        shifted = _check_square_matrix(
            empirical_normalized_shifted_covariance,
            "empirical_normalized_shifted_covariance",
        )
        if fc.shape != shifted.shape:
            raise ValueError(
                "empirical_fc and empirical_normalized_shifted_covariance "
                "must have the same shape."
            )
        fc = np.array(fc, dtype=float, copy=True)
        np.fill_diagonal(fc, 0.0)
        return LagcovObservables(
            fc=fc,
            normalized_shifted_covariance=shifted,
        )

    if empirical_timeseries is None:
        raise ValueError(
            "Provide either empirical_timeseries or both empirical_fc and "
            "empirical_normalized_shifted_covariance."
        )

    return average_lagcov_observables(
        empirical_timeseries,
        lag=lag,
        preprocess_fn=preprocess_fn,
    )


def _evaluate_lagcov_model(
    backend,
    gec,
    lag,
    tr_seconds,
    sigma,
    linear_model=None,
    check_stability=True,
    stability_tol=0.0,
    adapter_factory=None,
    theta=None,
    seeds=None,
    preprocess_fn=None,
):
    """Evaluate model FC and normalized shifted covariance for one GEC matrix."""
    if backend == "linear":
        if linear_model is None:
            raise ValueError("linear_model must be provided for backend='linear'.")
        evaluation = evaluate_linear_hopf(
            model=linear_model,
            weights=gec,
            sigma=sigma,
            lag_seconds=int(lag) * float(tr_seconds),
            check_stability=check_stability,
            tol=stability_tol,
        )
        return (
            LagcovObservables(
                fc=evaluation.fc,
                normalized_shifted_covariance=(
                    evaluation.normalized_shifted_covariance
                ),
            ),
            evaluation.max_real_eigenvalue,
        )

    if backend == "simulation":
        theta = dict(theta or {})
        theta.setdefault("G", 1.0)
        evaluation = simulate_lagcov_observables(
            adapter_factory=adapter_factory,
            weights=gec,
            theta=theta,
            seeds=seeds,
            lag=lag,
            preprocess_fn=preprocess_fn,
        )
        return evaluation.observables, np.nan

    raise ValueError("backend must be 'linear' or 'simulation'.")


def fit_lagcov_gec(
    sc,
    empirical_timeseries=None,
    empirical_fc=None,
    empirical_normalized_shifted_covariance=None,
    lag=1,
    tr_seconds=1.0,
    backend="linear",
    linear_model=None,
    sigma=0.02,
    adapter_factory=None,
    theta=None,
    seeds=None,
    preprocess_fn=None,
    init="scaled_sc",
    initial_gec=None,
    max_c=0.2,
    update_mask=None,
    include_homologue_edges=False,
    learning_rate_fc=0.0004,
    learning_rate_lag=0.0001,
    n_iter=1000,
    check_every=100,
    relative_tolerance=1e-4,
    stop_if_worse=True,
    allow_negative=False,
    l1_alpha=0.0,
    normalize_max=True,
    check_stability=True,
    stability_tol=0.0,
    mode="lagcov",
):
    """Fit lagcov-style Hopf generative effective connectivity.

    This implements the NEMO/Luppi-style update:
        C <- C + lr_fc * (FC_emp - FC_sim)
               + lr_lag * (Lag_emp - Lag_sim)
    """
    if mode != "lagcov":
        raise NotImplementedError(
            "Only mode='lagcov' is implemented. "
        )

    sc = _check_square_matrix(sc, "sc")
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be at least 1.")
    if n_iter < 1:
        raise ValueError("n_iter must be positive.")
    if check_every < 1:
        raise ValueError("check_every must be positive.")

    update_mask = _make_update_mask(
        sc=sc,
        update_mask=update_mask,
        include_homologue_edges=include_homologue_edges,
    )
    empirical = _prepare_empirical_observables(
        empirical_timeseries=empirical_timeseries,
        empirical_fc=empirical_fc,
        empirical_normalized_shifted_covariance=empirical_normalized_shifted_covariance,
        lag=lag,
        preprocess_fn=preprocess_fn,
    )
    if empirical.fc.shape != sc.shape:
        raise ValueError("Empirical observables must have the same shape as sc.")
    gec = _initialize_gec(
        sc=sc,
        update_mask=update_mask,
        init=init,
        initial_gec=initial_gec,
        max_c=max_c,
    )
    gec = _apply_constraints(
        gec=gec,
        update_mask=update_mask,
        allow_negative=allow_negative,
        l1_alpha=0.0,
        normalize_max=normalize_max,
        max_c=max_c,
    )

    loss_history = []
    max_real_history = []
    previous_checked_loss = None
    converged = False
    stop_reason = "maximum iterations reached"
    for iteration in range(int(n_iter)):
        simulated, max_real = _evaluate_lagcov_model(
            backend=backend,
            gec=gec,
            lag=lag,
            tr_seconds=tr_seconds,
            sigma=sigma,
            linear_model=linear_model,
            check_stability=check_stability,
            stability_tol=stability_tol,
            adapter_factory=adapter_factory,
            theta=theta,
            seeds=seeds,
            preprocess_fn=preprocess_fn,
        )
        max_real_history.append(max_real)

        fc_loss = _offdiag_mse(empirical.fc, simulated.fc)
        lag_loss = _offdiag_mse(
            empirical.normalized_shifted_covariance,
            simulated.normalized_shifted_covariance,
        )
        loss = fc_loss + lag_loss
        loss_history.append(loss)

        if iteration > 0 and iteration % int(check_every) == 0:
            if previous_checked_loss is not None:
                improvement = previous_checked_loss - loss
                if stop_if_worse and improvement < 0:
                    stop_reason = "loss increased at checkpoint"
                    break

                denominator = max(abs(loss), np.finfo(float).eps)
                relative_improvement = improvement / denominator
                if relative_improvement < float(relative_tolerance):
                    converged = True
                    stop_reason = "relative improvement below tolerance"
                    break

            previous_checked_loss = loss

        delta = (
            float(learning_rate_fc) * (empirical.fc - simulated.fc)
            + float(learning_rate_lag)
            * (
                empirical.normalized_shifted_covariance
                - simulated.normalized_shifted_covariance
            )
        )
        gec[update_mask] = gec[update_mask] + delta[update_mask]
        gec = _apply_constraints(
            gec=gec,
            update_mask=update_mask,
            allow_negative=allow_negative,
            l1_alpha=l1_alpha,
            normalize_max=normalize_max,
            max_c=max_c,
        )

    return LagcovGECResult(
        gec=gec,
        empirical_fc=empirical.fc,
        empirical_normalized_shifted_covariance=empirical.normalized_shifted_covariance,
        simulated_fc=simulated.fc,
        simulated_normalized_shifted_covariance=simulated.normalized_shifted_covariance,
        loss_history=loss_history,
        max_real_history=max_real_history,
        n_iter=len(loss_history),
        converged=converged,
        stop_reason=stop_reason,
    )
