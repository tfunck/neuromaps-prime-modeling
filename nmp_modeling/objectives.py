import numpy as np
from scipy.stats import ks_2samp

from nmp_modeling.observables import matrix_edges


def _paired_finite_vectors(a, b):
    """Return paired finite vectors with the same shape."""
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


def similarity(a, b, metric="pearson"):
    """Compute vector similarity using Pearson correlation or cosine similarity."""
    x, y = _paired_finite_vectors(a, b)

    if metric == "pearson":
        x_std = np.std(x)
        y_std = np.std(y)

        if x_std == 0 or y_std == 0:
            raise ValueError("Pearson correlation is undefined for constant vectors.")

        return float(np.corrcoef(x, y)[0, 1])

    if metric == "cosine":
        denom = np.linalg.norm(x) * np.linalg.norm(y)

        if denom == 0:
            raise ValueError("Cosine similarity is undefined for zero vectors.")

        return float(np.dot(x, y) / denom)

    raise ValueError("metric must be 'pearson' or 'cosine'.")


def similarity_distance(simulated, empirical, metric="pearson"):
    """Return negative similarity as a minimization distance."""
    return -similarity(simulated, empirical, metric=metric)


def edge_similarity_distance(simulated, empirical, triangle="upper", metric="pearson"):
    """Compare matrix edge patterns by Pearson or cosine similarity."""
    sim_edges = matrix_edges(simulated, triangle=triangle)
    emp_edges = matrix_edges(empirical, triangle=triangle)

    return similarity_distance(sim_edges, emp_edges, metric=metric)


def frobenius_distance(simulated, empirical):
    """Return Frobenius norm between two arrays."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.shape != emp.shape:
        raise ValueError("Inputs must have the same shape.")

    return float(np.linalg.norm(sim - emp))


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
    """Compare FC edge-value distributions by KS distance."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.ndim == 2 and sim.shape[0] == sim.shape[1]:
        sim = matrix_edges(sim, triangle=triangle)

    if emp.ndim == 2 and emp.shape[0] == emp.shape[1]:
        emp = matrix_edges(emp, triangle=triangle)

    return ks_distance(sim, emp)


def offdiag_mse_distance(simulated, empirical):
    """Return mean squared error over off-diagonal matrix entries."""
    sim = np.asarray(simulated, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    if sim.shape != emp.shape:
        raise ValueError("Inputs must have the same shape.")

    if (sim.ndim != 2 or sim.shape[0] != sim.shape[1]):
        raise ValueError("Inputs must be square 2D matrices.")

    mask = ~np.eye(sim.shape[0], dtype=bool)

    return float(np.mean((sim[mask] - emp[mask]) ** 2))
