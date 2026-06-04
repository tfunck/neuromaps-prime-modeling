import numpy as np


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
