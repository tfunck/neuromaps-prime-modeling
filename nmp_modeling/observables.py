import numpy as np


def as_time_by_nodes(data):
    """Return data as a 2D array with shape (time, nodes)."""
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        raise ValueError("Time series data must be a 2D array.")

    if arr.shape[0] < arr.shape[1]:
        arr = arr.T

    return arr


def matrix_edges(matrix, triangle="upper"):
    """Return off-diagonal matrix entries from the selected triangle."""
    mat = np.asarray(matrix, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    if triangle == "upper":
        idx = np.triu_indices_from(mat, k=1)
    elif triangle == "lower":
        idx = np.tril_indices_from(mat, k=-1)
    else:
        raise ValueError("triangle must be 'upper' or 'lower'.")

    return mat[idx]


def fisher_z_matrix(matrix, eps=1e-7):
    """Apply Fisher-z transform to a correlation-like matrix."""
    mat = np.array(matrix, dtype=float, copy=True)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    np.fill_diagonal(mat, 0.0)
    mat = np.clip(mat, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(mat)
    np.fill_diagonal(z, 0.0)

    return z


def inverse_fisher_z_matrix(matrix):
    """Apply inverse Fisher-z transform to a matrix."""
    mat = np.asarray(matrix, dtype=float)
    out = np.tanh(mat)

    if out.ndim == 2 and out.shape[0] == out.shape[1]:
        np.fill_diagonal(out, 0.0)

    return out


def compute_fc(data, method="pearson", fisher_z=False):
    """Compute a static FC-like matrix from BOLD time series."""
    ts = as_time_by_nodes(data)

    if method == "pearson":
        fc = np.corrcoef(ts.T)
    elif method == "covariance":
        if fisher_z:
            raise ValueError("Fisher-z is only supported for Pearson FC.")
        fc = np.cov(ts.T)
    else:
        raise ValueError("method must be 'pearson' or 'covariance'.")

    fc = np.asarray(fc, dtype=float)
    np.fill_diagonal(fc, 0.0)

    if fisher_z:
        fc = fisher_z_matrix(fc)

    return fc


def compute_gbc_from_fc(fc):
    """Compute node-level mean FC, also called GBC."""
    mat = np.asarray(fc, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("fc must be a square 2D array.")

    n_nodes = mat.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)

    return mat[mask].reshape(n_nodes, n_nodes - 1).mean(axis=1)


def compute_gbc(data, method="pearson", fisher_z=False):
    """Compute GBC from BOLD time series."""
    fc = compute_fc(data, method=method, fisher_z=fisher_z)
    return compute_gbc_from_fc(fc)


def compute_swfcd_distribution(
    data,
    window_size=30,
    step=2,
    fc_method="pearson",
    fisher_z=False,
    triangle="upper",
):
    """Compute a sliding-window FCD distribution from BOLD time series."""
    ts = as_time_by_nodes(data)
    n_time = ts.shape[0]

    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")

    if step <= 0:
        raise ValueError("step must be positive.")

    starts = np.arange(0, n_time - window_size + 1, step)

    if len(starts) < 2:
        raise ValueError("At least two windows are required to compute FCD.")

    fc_vectors = []

    for start in starts:
        window = ts[start:start + window_size]
        fc = compute_fc(window, method=fc_method, fisher_z=fisher_z)
        fc_vectors.append(matrix_edges(fc, triangle=triangle))

    fc_vectors = np.asarray(fc_vectors, dtype=float)
    fcd = np.corrcoef(fc_vectors)
    np.fill_diagonal(fcd, 0.0)

    return matrix_edges(fcd, triangle=triangle)


def compute_phfcd_distribution(data, triangle="upper"):
    """Compute a simple phase-FCD distribution from BOLD time series."""
    from scipy.signal import hilbert

    ts = as_time_by_nodes(data)
    phase = np.angle(hilbert(ts, axis=0))

    phase_vectors = []

    for t in range(phase.shape[0]):
        delta = phase[t, :, None] - phase[t, None, :]
        coherence = np.cos(delta)
        np.fill_diagonal(coherence, 0.0)
        phase_vectors.append(matrix_edges(coherence, triangle=triangle))

    phase_vectors = np.asarray(phase_vectors, dtype=float)

    if phase_vectors.shape[0] < 2:
        raise ValueError("At least two time points are required to compute phFCD.")

    phfcd = np.corrcoef(phase_vectors)
    np.fill_diagonal(phfcd, 0.0)

    return matrix_edges(phfcd, triangle=triangle)


# Backward-compatible aliases for the original public functions.

def fc_fisher_z(bold):
    """Backward-compatible alias for Fisher-z-transformed Pearson FC."""
    return compute_fc(bold, method="pearson", fisher_z=True)


def swfcd(bold, window_size=30, step=3):
    """Backward-compatible alias for sliding-window FCD."""
    return compute_swfcd_distribution(
        bold,
        window_size=window_size,
        step=step,
        fc_method="pearson",
        fisher_z=False,
        triangle="upper",
    )


def pearson_lower_triangle(a, b):
    """Backward-compatible FC lower-triangle correlation distance."""
    a_edges = matrix_edges(a, triangle="lower")
    b_edges = matrix_edges(b, triangle="lower")
    r = np.corrcoef(a_edges, b_edges)[0, 1]
    return -float(r)


def ks_distance(a, b):
    """Backward-compatible KS distance."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(np.ravel(a), np.ravel(b)).statistic)