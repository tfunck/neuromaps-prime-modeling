import numpy as np
from dataclasses import dataclass

from nmp_modeling.objectives import offdiag_mse_distance
from nmp_modeling.hopf.linear import evaluate_linear_hopf
from nmp_modeling.hopf.simulation import (
    LagcovObservables,
    MIObservables,
    average_lagcov_observables,
    average_mi_observables,
    simulate_lagcov_observables,
    simulate_mi_observables,
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
    best_loss: float
    best_iter: int
    fc_corr_history: list | None = None
    best_fc_corr: float | None = None
    best_fc_corr_iter: int | None = None
    best_fc_corr_gec: np.ndarray | None = None


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


def _safe_corr(a, b):
    """Return Pearson correlation, or NaN if one vector is constant."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("Correlation inputs must have the same size.")
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _offdiag_corr(a, b):
    """Return off-diagonal Pearson correlation between two square matrices."""
    return _safe_corr(_offdiag_values(a), _offdiag_values(b))


def _check_finite_loss(loss):
    """Validate and return a finite scalar loss."""
    loss = float(loss)
    if not np.isfinite(loss):
        raise ValueError(f"Loss is not finite: {loss}.")
    return loss


def _copy_lagcov_observables(observables):
    """Copy lagcov observables so the best state remains immutable."""
    return LagcovObservables(
        fc=np.array(observables.fc, dtype=float, copy=True),
        normalized_shifted_covariance=np.array(
            observables.normalized_shifted_covariance,
            dtype=float,
            copy=True,
        ),
    )


def _copy_mi_observables(observables):
    """Copy MI observables so the best state remains immutable."""
    return MIObservables(
        fc_mi=np.array(observables.fc_mi, dtype=float, copy=True),
        forward_mi=np.array(observables.forward_mi, dtype=float, copy=True),
        reverse_mi=np.array(observables.reverse_mi, dtype=float, copy=True),
    )


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


def _cap_to_max_c(gec, update_mask, max_c, allow_negative=False):
    """Clip updated GEC weights to the allowed max-C range."""
    out = np.array(gec, dtype=float, copy=True)
    limit = abs(float(max_c))
    lower = -limit if allow_negative else 0.0
    out[update_mask] = np.clip(out[update_mask], lower, limit)
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
    cap_max=True,
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
    if cap_max:
        out = _cap_to_max_c(
            gec=out,
            update_mask=update_mask,
            max_c=max_c,
            allow_negative=allow_negative,
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
    monitor_fc_corr=False,
    allow_negative=False,
    l1_alpha=0.0,
    cap_max=True,
    check_stability=True,
    stability_tol=0.0,
):
    """Fit lagcov-style Hopf generative effective connectivity.

    This implements the NEMO/Luppi-style update:
        C <- C + lr_fc * (FC_emp - FC_sim)
               + lr_lag * (Lag_emp - Lag_sim)
    """
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
        cap_max=cap_max,
        max_c=max_c,
    )

    loss_history = []
    max_real_history = []
    previous_checked_loss = None
    converged = False
    stop_reason = "maximum iterations reached"
    best_loss = np.inf
    best_iter = -1
    fc_corr_history = [] if monitor_fc_corr else None
    previous_checked_fc_corr = None
    best_fc_corr = None
    best_fc_corr_iter = None
    best_fc_corr_gec = None
    for iteration in range(int(n_iter)):
        evaluated_gec = np.array(gec, dtype=float, copy=True)

        try:
            simulated, max_real = _evaluate_lagcov_model(
                backend=backend,
                gec=evaluated_gec,
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

            fc_loss = offdiag_mse_distance(empirical.fc, simulated.fc)
            lag_loss = offdiag_mse_distance(
                empirical.normalized_shifted_covariance,
                simulated.normalized_shifted_covariance,
            )
            loss = _check_finite_loss(fc_loss + lag_loss)
            fc_corr = None
            if monitor_fc_corr:
                fc_corr = _offdiag_corr(empirical.fc, simulated.fc)

        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            stop_reason = f"invalid model evaluation at iteration {iteration}: {exc}"
            converged = False
            break

        max_real_history.append(max_real)
        loss_history.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_iter = iteration
            best_gec = evaluated_gec
            best_simulated = _copy_lagcov_observables(simulated)

        if monitor_fc_corr:
            fc_corr_history.append(fc_corr)
            if (
                fc_corr is not None
                and np.isfinite(fc_corr)
                and (best_fc_corr is None or fc_corr > best_fc_corr)
            ):
                best_fc_corr = fc_corr
                best_fc_corr_iter = iteration
                best_fc_corr_gec = np.array(evaluated_gec, dtype=float, copy=True)

        if iteration == 0:
            previous_checked_loss = loss
            if monitor_fc_corr:
                previous_checked_fc_corr = fc_corr

        elif iteration % int(check_every) == 0:
            improvement = previous_checked_loss - loss
            loss_increased = improvement < 0.0
            skip_tolerance_check = False

            if stop_if_worse and loss_increased:
                if monitor_fc_corr:
                    fc_corr_decreased = (
                        fc_corr is not None
                        and previous_checked_fc_corr is not None
                        and np.isfinite(fc_corr)
                        and np.isfinite(previous_checked_fc_corr)
                        and fc_corr < previous_checked_fc_corr
                    )

                    if fc_corr_decreased:
                        stop_reason = (
                            "loss increased and FC correlation decreased at checkpoint"
                        )
                        break

                    # If loss worsens but FC correlation does not worsen,
                    # keep going instead of treating the negative improvement
                    # as convergence.
                    skip_tolerance_check = True

                else:
                    stop_reason = "loss increased at checkpoint"
                    break

            if relative_tolerance is not None and not skip_tolerance_check:
                denominator = max(abs(loss), np.finfo(float).eps)
                relative_improvement = improvement / denominator
                if relative_improvement < float(relative_tolerance):
                    converged = True
                    stop_reason = "relative improvement below tolerance"
                    break

            previous_checked_loss = loss
            if monitor_fc_corr:
                previous_checked_fc_corr = fc_corr

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
            cap_max=cap_max,
            max_c=max_c,
        )

    return LagcovGECResult(
        gec=best_gec,
        empirical_fc=empirical.fc,
        empirical_normalized_shifted_covariance=empirical.normalized_shifted_covariance,
        simulated_fc=best_simulated.fc,
        simulated_normalized_shifted_covariance=best_simulated.normalized_shifted_covariance,
        loss_history=loss_history,
        max_real_history=max_real_history,
        n_iter=len(loss_history),
        converged=converged,
        stop_reason=stop_reason,
        best_loss=best_loss,
        best_iter=best_iter,
        fc_corr_history=fc_corr_history,
        best_fc_corr=best_fc_corr,
        best_fc_corr_iter=best_fc_corr_iter,
        best_fc_corr_gec=best_fc_corr_gec,
    )


@dataclass
class MIGECResult:
    """Container for MI/NR GEC fitting results."""
    gec: np.ndarray
    empirical_fc_mi: np.ndarray
    empirical_forward_mi: np.ndarray
    empirical_reverse_mi: np.ndarray
    simulated_fc_mi: np.ndarray
    simulated_forward_mi: np.ndarray
    simulated_reverse_mi: np.ndarray
    loss_history: list
    n_iter: int
    converged: bool
    stop_reason: str
    best_loss: float
    best_iter: int


def _prepare_mi_empirical_observables(
    empirical_timeseries=None,
    empirical_fc_mi=None,
    empirical_forward_mi=None,
    empirical_reverse_mi=None,
    lag=2,
    preprocess_fn=None,
    eps=1e-12,
):
    """Create empirical MI/NR targets."""
    provided = [
        empirical_fc_mi is not None,
        empirical_forward_mi is not None,
        empirical_reverse_mi is not None,
    ]
    if any(provided) and not all(provided):
        raise ValueError(
            "Provide all of empirical_fc_mi, empirical_forward_mi, and "
            "empirical_reverse_mi, or provide empirical_timeseries."
        )
    if all(provided):
        fc_mi = _check_square_matrix(empirical_fc_mi, "empirical_fc_mi")
        forward_mi = _check_square_matrix(empirical_forward_mi, "empirical_forward_mi")
        reverse_mi = _check_square_matrix(empirical_reverse_mi, "empirical_reverse_mi")
        if fc_mi.shape != forward_mi.shape or fc_mi.shape != reverse_mi.shape:
            raise ValueError("All empirical MI matrices must have the same shape.")
        fc_mi = np.array(fc_mi, dtype=float, copy=True)
        forward_mi = np.array(forward_mi, dtype=float, copy=True)
        reverse_mi = np.array(reverse_mi, dtype=float, copy=True)
        np.fill_diagonal(fc_mi, 0.0)
        np.fill_diagonal(forward_mi, 0.0)
        np.fill_diagonal(reverse_mi, 0.0)
        return MIObservables(
            fc_mi=fc_mi,
            forward_mi=forward_mi,
            reverse_mi=reverse_mi,
        )

    if empirical_timeseries is None:
        raise ValueError(
            "Provide empirical_timeseries or all three precomputed empirical "
            "MI matrices."
        )

    return average_mi_observables(
        empirical_timeseries,
        lag=lag,
        preprocess_fn=preprocess_fn,
        eps=eps,
    )


def _evaluate_mi_model(
    gec,
    lag,
    adapter_factory,
    theta,
    seeds,
    preprocess_fn=None,
    eps=1e-12,
):
    """Evaluate simulated MI/NR observables for one GEC matrix."""
    theta = dict(theta or {})
    theta.setdefault("G", 1.0)
    evaluation = simulate_mi_observables(
        adapter_factory=adapter_factory,
        weights=gec,
        theta=theta,
        seeds=seeds,
        lag=lag,
        preprocess_fn=preprocess_fn,
        eps=eps,
    )
    return evaluation.observables


def fit_mi_nr_gec(
    sc,
    empirical_timeseries=None,
    empirical_fc_mi=None,
    empirical_forward_mi=None,
    empirical_reverse_mi=None,
    lag=2,
    adapter_factory=None,
    theta=None,
    seeds=None,
    preprocess_fn=None,
    init="zeros",
    initial_gec=None,
    max_c=0.2,
    update_mask=None,
    include_homologue_edges=True,
    learning_rate_fc=0.0005,
    learning_rate_nr=0.0001,
    use_reversal=True,
    n_iter=3000,
    check_every=100,
    relative_tolerance=None,
    stop_if_worse=False,
    allow_negative=False,
    eps=1e-12,
):
    """Fit MI/NR-style Hopf generative effective connectivity.

    The update is:
        C <- C
             + lr_fc * (I_fc_emp - I_fc_sim)
             - lr_nr * ((I_forward_emp - I_reverse_emp) - (I_forward_sim - I_reverse_sim))
    """
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
    empirical = _prepare_mi_empirical_observables(
        empirical_timeseries=empirical_timeseries,
        empirical_fc_mi=empirical_fc_mi,
        empirical_forward_mi=empirical_forward_mi,
        empirical_reverse_mi=empirical_reverse_mi,
        lag=lag,
        preprocess_fn=preprocess_fn,
        eps=eps,
    )
    if empirical.fc_mi.shape != sc.shape:
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
        max_c=max_c,
    )

    loss_history = []
    previous_checked_loss = None
    converged = False
    stop_reason = "maximum iterations reached"
    best_loss = np.inf
    best_iter = -1
    for iteration in range(int(n_iter)):
        evaluated_gec = np.array(gec, dtype=float, copy=True)

        try:
            simulated = _evaluate_mi_model(
                gec=evaluated_gec,
                lag=lag,
                adapter_factory=adapter_factory,
                theta=theta,
                seeds=seeds,
                preprocess_fn=preprocess_fn,
                eps=eps,
            )

            fc_loss = offdiag_mse_distance(empirical.fc_mi, simulated.fc_mi)
            if use_reversal:
                empirical_nr = empirical.forward_mi - empirical.reverse_mi
                simulated_nr = simulated.forward_mi - simulated.reverse_mi
                nr_loss = offdiag_mse_distance(empirical_nr, simulated_nr)
                loss = fc_loss + nr_loss
            else:
                loss = fc_loss
            loss = _check_finite_loss(loss)

        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            stop_reason = f"invalid model evaluation at iteration {iteration}: {exc}"
            converged = False
            break

        loss_history.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_iter = iteration
            best_gec = evaluated_gec
            best_simulated = _copy_mi_observables(simulated)

        if iteration == 0:
            previous_checked_loss = loss
        elif iteration % int(check_every) == 0:
            improvement = previous_checked_loss - loss
            if stop_if_worse and improvement < 0:
                stop_reason = "loss increased at checkpoint"
                break
            if relative_tolerance is not None:
                denominator = max(abs(loss), np.finfo(float).eps)
                relative_improvement = improvement / denominator
                if relative_improvement < float(relative_tolerance):
                    converged = True
                    stop_reason = "relative improvement below tolerance"
                    break

            previous_checked_loss = loss

        delta = float(learning_rate_fc) * (empirical.fc_mi - simulated.fc_mi)
        if use_reversal:
            empirical_nr = empirical.forward_mi - empirical.reverse_mi
            simulated_nr = simulated.forward_mi - simulated.reverse_mi
            nr_residual = empirical_nr - simulated_nr
            delta = delta - float(learning_rate_nr) * nr_residual
        gec[update_mask] = gec[update_mask] + delta[update_mask]
        gec = _apply_constraints(
            gec=gec,
            update_mask=update_mask,
            allow_negative=allow_negative,
            l1_alpha=0.0,
            max_c=max_c,
        )

    return MIGECResult(
        gec=best_gec,
        empirical_fc_mi=empirical.fc_mi,
        empirical_forward_mi=empirical.forward_mi,
        empirical_reverse_mi=empirical.reverse_mi,
        simulated_fc_mi=best_simulated.fc_mi,
        simulated_forward_mi=best_simulated.forward_mi,
        simulated_reverse_mi=best_simulated.reverse_mi,
        loss_history=loss_history,
        n_iter=len(loss_history),
        converged=converged,
        stop_reason=stop_reason,
        best_loss=best_loss,
        best_iter=best_iter,
    )
