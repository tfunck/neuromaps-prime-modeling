import numpy as np

def check_time_series(data):
    """Validate time series data with shape (time, nodes)."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Time series data must be a 2D array with shape (time, nodes).")
    if arr.shape[0] < 2:
        raise ValueError("Time series data must contain at least two time points.")
    if arr.shape[1] < 2:
        raise ValueError("Time series data must contain at least two nodes.")
    return arr

def check_square_matrix(matrix, name="matrix"):
    """Validate a square 2D matrix."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix.")
    return mat

def matrix_edges(matrix, triangle="upper"):
    """Return off-diagonal entries from one triangle of a square matrix."""
    mat = check_square_matrix(matrix)
    if triangle == "upper":
        idx = np.triu_indices_from(mat, k=1)
    elif triangle == "lower":
        idx = np.tril_indices_from(mat, k=-1)
    else:
        raise ValueError("triangle must be 'upper' or 'lower'.")
    return mat[idx]

def fisher_z_matrix(matrix, eps=1e-7):
    """Apply Fisher-z transform to a correlation matrix."""
    mat = np.array(check_square_matrix(matrix), dtype=float, copy=True)
    np.fill_diagonal(mat, 0.0)
    mat = np.clip(mat, -1.0 + eps, 1.0 - eps)
    return np.arctanh(mat)

def inverse_fisher_z_matrix(matrix):
    """Apply inverse Fisher-z transform to a matrix."""
    return np.tanh(np.asarray(matrix, dtype=float))

def compute_fc(timeseries, fisher_z=False):
    """Compute Pearson FC from time series with shape (time, nodes)."""
    ts = check_time_series(timeseries)
    fc = np.asarray(np.corrcoef(ts.T), dtype=float)
    np.fill_diagonal(fc, 0.0)
    if fisher_z:
        fc = fisher_z_matrix(fc)
    return fc

def compute_gbc_from_fc(fc):
    """Compute node-level mean FC, also known as GBC."""
    mat = check_square_matrix(fc, name="fc")
    n_nodes = mat.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)
    values = mat[mask].reshape(n_nodes, n_nodes - 1)
    return values.mean(axis=1)

def compute_gbc(timeseries, fisher_z=False):
    """Compute GBC from time series."""
    fc = compute_fc(timeseries, fisher_z=fisher_z)
    return compute_gbc_from_fc(fc)

def compute_swfcd_distribution(
    timeseries,
    window_size=30,
    step=2,
    fisher_z=False,
    triangle="upper",
):
    """Compute sliding-window FCD distribution from time series."""
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if step <= 0:
        raise ValueError("step must be positive.")

    ts = check_time_series(timeseries)
    n_time = ts.shape[0]
    starts = np.arange(0, n_time - window_size + 1, step)
    if len(starts) < 2:
        raise ValueError("At least two windows are required to compute FCD.")

    window_fc_edges = []
    for start in starts:
        window = ts[start:start + window_size]
        fc = compute_fc(window, fisher_z=fisher_z)
        window_fc_edges.append(matrix_edges(fc, triangle=triangle))

    window_fc_edges = np.asarray(window_fc_edges, dtype=float)
    fcd = np.corrcoef(window_fc_edges)
    return matrix_edges(fcd, triangle=triangle)

def compute_phfcd_distribution(
    timeseries,
    discard_offset=10,
    pattern_size=3,
    triangle="upper",
):
    """Compute phase-FCD distribution from time series."""
    from scipy.signal import hilbert
    if pattern_size < 1:
        raise ValueError("pattern_size must be at least 1.")
    if discard_offset < 0:
        raise ValueError("discard_offset must be non-negative.")

    ts = check_time_series(timeseries)
    n_time = ts.shape[0]
    start = discard_offset
    stop = n_time - discard_offset + 1
    if stop <= start:
        raise ValueError("Time series is too short for the requested discard_offset.")

    ts = ts - np.mean(ts, axis=0, keepdims=True)
    phase = np.angle(hilbert(ts, axis=0))[start:stop]
    if phase.shape[0] < pattern_size + 1:
        raise ValueError("Not enough phase samples for the requested pattern_size.")

    edges = []
    for t in range(phase.shape[0]):
        delta = phase[t, :, None] - phase[t, None, :]
        edges.append(matrix_edges(np.cos(delta), triangle=triangle))
    edges = np.asarray(edges, dtype=float)
    patterns = np.asarray(
        [edges[i:i + pattern_size].sum(axis=0)
         for i in range(edges.shape[0] - pattern_size + 1)],
        dtype=float,
    )

    values = []
    norms = np.linalg.norm(patterns, axis=1)
    for i in range(patterns.shape[0]):
        for j in range(i + 1, patterns.shape[0]):
            denom = norms[i] * norms[j]
            values.append(0.0 if denom == 0 else np.dot(patterns[i], patterns[j]) / denom)

    return np.asarray(values, dtype=float)
