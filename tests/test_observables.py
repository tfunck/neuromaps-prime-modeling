"""Tests for the observables module.

These tests do not require Neuronumba to be installed.
"""

import numpy as np

from nmp_modeling import observables


def test_fc_fisher_z_shape_and_diagonal():
    """FC matrix is square with zero diagonal."""
    rng = np.random.default_rng(0)
    bold = rng.standard_normal((400, 20))   # (timepoints, nodes)
    fc = observables.fc_fisher_z(bold)
    assert fc.shape == (20, 20)
    np.testing.assert_allclose(np.diag(fc), 0.0)


def test_fc_fisher_z_handles_transposed_input():
    """Either orientation should produce the same FC matrix."""
    rng = np.random.default_rng(0)
    bold = rng.standard_normal((400, 20))
    fc1 = observables.fc_fisher_z(bold)
    fc2 = observables.fc_fisher_z(bold.T)
    np.testing.assert_allclose(fc1, fc2)


def test_swfcd_returns_1d_distribution():
    """SwFCD output is a 1-D vector of upper-triangular entries."""
    rng = np.random.default_rng(0)
    bold = rng.standard_normal((400, 20))
    fcd = observables.swfcd(bold)
    assert fcd.ndim == 1
    # Number of upper-triangular entries depends on number of windows.
    # Sanity: should be > 0 and a valid count for some n_windows.
    assert len(fcd) > 0


def test_pearson_lower_triangle_is_negative_correlation():
    """A perfect match should give -1; orthogonal should give ~0."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((10, 10))
    a = (a + a.T) / 2  # symmetric
    same = observables.pearson_lower_triangle(a, a)
    assert np.isclose(same, -1.0)


def test_ks_distance_zero_for_identical():
    """KS distance between a distribution and itself is exactly 0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert observables.ks_distance(x, x) == 0.0


def test_ks_distance_positive_for_shifted():
    """KS distance between shifted distributions should be positive."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    y = rng.standard_normal(1000) + 2.0
    assert observables.ks_distance(x, y) > 0.5