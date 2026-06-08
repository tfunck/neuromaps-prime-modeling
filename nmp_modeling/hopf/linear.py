import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def _check_square_matrix(matrix, name):
    """Validate that matrix is a square 2D array."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return arr


def max_real_eigenvalue(jacobian):
    """Return the largest real part of the eigenvalues of a Jacobian."""
    A = _check_square_matrix(jacobian, "jacobian")
    eigvals = np.linalg.eigvals(A)
    return float(np.max(eigvals.real))


def is_stable_jacobian(jacobian, tol=0.0):
    """Return whether a linear system is asymptotically stable."""
    return max_real_eigenvalue(jacobian) < -float(tol)


def check_stable_jacobian(jacobian, tol=0.0):
    """Raise an error if a Jacobian is not asymptotically stable."""
    max_real = max_real_eigenvalue(jacobian)
    if max_real >= -float(tol):
        raise ValueError(
            "Jacobian is not asymptotically stable: "
            f"max real eigenvalue = {max_real}."
        )
    return max_real


def make_noise_covariance(sigma, n_nodes):
    """Create a 2N-by-2N diagonal noise covariance matrix for Hopf states."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")

    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        sigmas = np.full(2 * n_nodes, float(arr), dtype=float)
    elif arr.ndim == 1:
        if arr.size == 1:
            sigmas = np.full(2 * n_nodes, float(arr[0]), dtype=float)
        elif arr.size == n_nodes:
            sigmas = np.concatenate([arr, arr]).astype(float, copy=False)
        elif arr.size == 2 * n_nodes:
            sigmas = arr.astype(float, copy=False)
        else:
            raise ValueError(
                "1D sigma must have length 1, n_nodes, or 2 * n_nodes."
            )
    elif arr.ndim == 2:
        if arr.shape == (2, n_nodes):
            sigmas = np.concatenate([arr[0], arr[1]]).astype(float, copy=False)
        elif arr.shape == (2 * n_nodes, 2 * n_nodes):
            return arr.astype(float, copy=False)
        else:
            raise ValueError(
                "2D sigma must have shape (2, n_nodes) or "
                "(2 * n_nodes, 2 * n_nodes)."
            )
    else:
        raise ValueError("sigma must be a scalar, vector, or matrix.")

    if np.any(sigmas < 0):
        raise ValueError("Noise standard deviations must be non-negative.")

    return np.diag(sigmas * sigmas)


def solve_stationary_covariance(
    jacobian,
    noise_covariance,
    check_stability=True,
    tol=0.0,
):
    """Solve A C + C A.T + Q = 0 for the stationary covariance C."""
    A = _check_square_matrix(jacobian, "jacobian")
    Q = _check_square_matrix(noise_covariance, "noise_covariance")
    if A.shape != Q.shape:
        raise ValueError("jacobian and noise_covariance must have the same shape.")
    if check_stability:
        check_stable_jacobian(A, tol=tol)

    cov = solve_continuous_lyapunov(A, -Q)
    return 0.5 * (cov + cov.T)


def covariance_to_correlation(covariance):
    """Convert a covariance matrix to a correlation matrix."""
    cov = _check_square_matrix(covariance, "covariance")
    var = np.diag(cov)
    if np.any(var <= 0):
        raise ValueError("Cannot convert covariance with non-positive variances.")
    scale = np.sqrt(var[:, None] * var[None, :])
    return cov / scale


def linear_fc_from_covariance(covariance, n_nodes):
    """Return the x-state FC block from a full Hopf covariance matrix."""
    return covariance_to_correlation(covariance[:n_nodes, :n_nodes])


def linear_shifted_covariance(jacobian, covariance, lag_seconds):
    """Compute C(tau) = exp(A * tau) C(0)."""
    A = _check_square_matrix(jacobian, "jacobian")
    cov = _check_square_matrix(covariance, "covariance")
    if A.shape != cov.shape:
        raise ValueError("jacobian and covariance must have the same shape.")
    if lag_seconds < 0:
        raise ValueError("lag_seconds must be non-negative.")
    return expm(float(lag_seconds) * A) @ cov


def linear_normalized_shifted_covariance(jacobian, covariance, n_nodes, lag_seconds):
    """Return the normalized x-state shifted covariance block."""
    shifted = linear_shifted_covariance(
        jacobian=jacobian,
        covariance=covariance,
        lag_seconds=lag_seconds,
    )
    shifted_x = shifted[:n_nodes, :n_nodes]
    zero_var = np.diag(covariance[:n_nodes, :n_nodes])
    if np.any(zero_var <= 0):
        raise ValueError("Cannot normalize shifted covariance with zero variance.")
    scale = np.sqrt(zero_var[:, None] * zero_var[None, :])
    return shifted_x / scale
