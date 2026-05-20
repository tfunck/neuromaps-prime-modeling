import numpy as np
from scipy.stats import ks_2samp

from nmp_modeling.observables import matrix_edges


def _finite_vectors(a, b):
    """Return finite paired vectors."""
    x = np.ravel(np.asarray(a, dtype=float))
    y = np.ravel(np.asarray(b, dtype=float))

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise ValueError("At least two finite paired values are required.")

    return x, y


def correlation(a, b):
    """Compute Pearson correlation between two finite vectors."""
    x, y = _finite_vectors(a, b)

    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("Correlation is undefined for constant vectors.")

    return float(np.corrcoef(x, y)[0, 1])


def vector_correlation_distance(simulated, empirical):
    """Return negative Pearson correlation as a minimization distance."""
    return -correlation(simulated, empirical)


def edge_correlation_distance(simulated, empirical, triangle="upper"):
    """Return negative edge-wise FC correlation."""
    sim_edges = matrix_edges(simulated, triangle=triangle)
    emp_edges = matrix_edges(empirical, triangle=triangle)

    return vector_correlation_distance(sim_edges, emp_edges)


def frobenius_distance(simulated, empirical):
    """Return Frobenius norm between two matrices."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.shape != emp.shape:
        raise ValueError("Inputs must have the same shape.")

    return float(np.linalg.norm(sim - emp, ord="fro"))


def mse_distance(simulated, empirical):
    """Return mean squared error between two arrays."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.shape != emp.shape:
        raise ValueError("Inputs must have the same shape.")

    return float(np.mean((sim - emp) ** 2))


def mean_abs_difference(simulated, empirical, triangle="upper"):
    """Return absolute difference between mean values."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.ndim == 2 and sim.shape[0] == sim.shape[1]:
        sim = matrix_edges(sim, triangle=triangle)

    if emp.ndim == 2 and emp.shape[0] == emp.shape[1]:
        emp = matrix_edges(emp, triangle=triangle)

    return float(abs(np.mean(sim) - np.mean(emp)))


def ks_distance(simulated, empirical):
    """Return two-sample Kolmogorov-Smirnov distance."""
    sim = np.ravel(np.asarray(simulated, dtype=float))
    emp = np.ravel(np.asarray(empirical, dtype=float))

    sim = sim[np.isfinite(sim)]
    emp = emp[np.isfinite(emp)]

    if sim.size == 0 or emp.size == 0:
        raise ValueError("KS distance requires non-empty finite samples.")

    return float(ks_2samp(sim, emp).statistic)


def fc_distribution_ks_distance(simulated, empirical, triangle="upper"):
    """Return KS distance between FC edge-value distributions."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.ndim == 2 and sim.shape[0] == sim.shape[1]:
        sim = matrix_edges(sim, triangle=triangle)

    if emp.ndim == 2 and emp.shape[0] == emp.shape[1]:
        emp = matrix_edges(emp, triangle=triangle)

    return ks_distance(sim, emp)


# Backward-compatible alias.

def pearson_lower_triangle(a, b):
    """Return negative lower-triangle FC correlation."""
    return edge_correlation_distance(a, b, triangle="lower")