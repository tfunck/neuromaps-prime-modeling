"""Layer 3a: backend-agnostic observables and distance functions.

These are plain functions, not a class hierarchy. The fitting layer
accepts any callable that takes BOLD ``(n_timepoints, n_nodes)`` and returns
something — a matrix, a vector, a scalar — that the user knows how to
compare with their empirical target via the supplied distance function.

We provide a small set of common defaults; users can pass their own
callables for anything else.
"""

from __future__ import annotations

import numpy as np


def _fisher_z(fc: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Fisher-z transform of an FC matrix, with diagonal zeroed."""
    x = np.array(fc, dtype=float, copy=True)
    np.fill_diagonal(x, 0.0)
    x = np.clip(x, -1.0 + eps, 1.0 - eps)
    return np.arctanh(x)


def fc_fisher_z(bold: np.ndarray) -> np.ndarray:
    """Static functional connectivity, Fisher-z transformed.

    Parameters
    ----------
    bold : np.ndarray
        Shape ``(n_timepoints, n_nodes)`` *or* ``(n_nodes, n_timepoints)``.
        We assume the longer axis is time and transpose if necessary.

    Returns
    -------
    np.ndarray
        ``(n_nodes, n_nodes)`` Fisher-z FC matrix. Diagonal is zero.
    """
    bold = np.asarray(bold)
    # Heuristic: timepoints almost always > nodes for fMRI.
    if bold.shape[0] < bold.shape[1]:
        bold = bold.T
    fc = np.corrcoef(bold.T)
    return _fisher_z(fc)


def swfcd(
    bold: np.ndarray,
    window_size: int = 30,
    step: int = 3,
) -> np.ndarray:
    """Sliding-window functional connectivity dynamics, returned as a 1-D
    sample of the off-diagonal entries.

    The standard FCD matrix is the correlation between FC matrices at
    different time windows. Following Deco et al. and the convention used
    in Neuronumba's ``SwFCD`` observable, what gets compared between two
    FCD matrices is the *distribution* of off-diagonal entries (via KS
    distance, since there's no temporal correspondence between subjects).
    So we return a 1-D vector of those entries directly — this is what the
    KS distance wants as input.

    Parameters
    ----------
    bold : np.ndarray
        ``(n_timepoints, n_nodes)`` or ``(n_nodes, n_timepoints)``.
    window_size : int
        Number of timepoints per sliding window. Defaults to 30 (~60s at
        TR=2s, the convention in Zhang et al. 2024 and Deco 2018).
    step : int
        Stride between successive windows.

    Returns
    -------
    np.ndarray
        1-D array of upper-triangular FCD matrix entries.
    """
    bold = np.asarray(bold)
    if bold.shape[0] < bold.shape[1]:
        bold = bold.T  # now (n_timepoints, n_nodes)

    n_t, n_nodes = bold.shape
    starts = np.arange(0, n_t - window_size + 1, step)

    # Vectorise the FC computation across windows. For a 200-window run
    # this is much faster than a python loop.
    triu_idx = np.triu_indices(n_nodes, k=1)
    window_fcs = np.empty((len(starts), len(triu_idx[0])))
    for i, s in enumerate(starts):
        w = bold[s:s + window_size]
        # Pearson correlation across nodes within this window.
        fc = np.corrcoef(w.T)
        window_fcs[i] = fc[triu_idx]

    # FCD = correlation between flattened windowed-FC vectors.
    fcd = np.corrcoef(window_fcs)

    # Return the upper-triangular entries as a 1-D distribution.
    return fcd[np.triu_indices_from(fcd, k=1)]


def pearson_lower_triangle(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between the strict lower triangles of two
    square matrices. Useful as a *similarity* measure for FC matrices.

    Returns the negative correlation so that smaller is better, matching
    the convention of the other distances in this module. (We want
    "minimise loss", not "maximise fit", so all distances minimise.)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    idx = np.tril_indices_from(a, k=-1)
    r = np.corrcoef(a[idx], b[idx])[0, 1]
    return -float(r)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between two 1-D distributions.

    This is the standard FCD comparison metric (Hansen et al. 2015,
    Deco 2018, Zhang 2024) precisely because FCD distributions don't have
    temporal correspondence between simulated and empirical data — we
    compare distributions, not pointwise.
    """
    from scipy.stats import ks_2samp
    return float(ks_2samp(np.ravel(a), np.ravel(b)).statistic)