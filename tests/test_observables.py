"""Tests for observables and distance functions."""

import numpy as np

from nmp_modeling import observables, objectives


def _bold(n_timepoints=300, n_nodes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_timepoints, n_nodes))


def test_compute_fc_shape_and_diagonal():
    bold = _bold()
    fc = observables.compute_fc(bold)
    assert fc.shape == (10, 10)
    assert np.allclose(np.diag(fc), 0.0)  # diagonal is zeroed by design
    assert np.allclose(fc, fc.T)


def test_compute_fc_fisher_z_finite_and_symmetric():
    bold = _bold()
    zfc = observables.compute_fc(bold, fisher_z=True)
    assert zfc.shape == (10, 10)
    assert np.all(np.isfinite(zfc))
    assert np.allclose(zfc, zfc.T)


def test_swfcd_returns_1d_distribution():
    bold = _bold(n_timepoints=400)
    fcd = observables.compute_swfcd_distribution(bold, window_size=30, step=3)
    assert fcd.ndim == 1
    assert fcd.size > 0
    assert np.all(fcd >= -1.0) and np.all(fcd <= 1.0)


def test_matrix_edges_extracts_off_diagonal():
    m = np.arange(16, dtype=float).reshape(4, 4)
    upper = observables.matrix_edges(m, triangle="upper")
    assert upper.size == 6
    assert 0.0 not in upper  # diagonal excluded


def test_edge_similarity_distance_is_minus_one_for_identical():
    bold = _bold()
    fc = observables.compute_fc(bold)
    # distance = -Pearson(edges), so identical matrices give -1
    assert np.isclose(objectives.edge_similarity_distance(fc, fc), -1.0)


def test_ks_distance_zero_for_identical():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000)
    assert objectives.ks_distance(x, x) == 0.0


def test_ks_distance_positive_for_shifted():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1000)
    y = rng.standard_normal(1000) + 2.0
    assert objectives.ks_distance(x, y) > 0.5
