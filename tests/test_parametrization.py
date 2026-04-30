"""Tests for the MapParametrization layer.

These tests do not require Neuronumba to be installed.
"""

import numpy as np
import pytest

from nmp_modeling.parametrization import MapParametrization, FreeParam


def test_linear_evaluation():
    """Basic case: a + b * gaba — verify against numpy."""
    gaba = np.array([0.1, 0.5, 0.7, 0.3, 0.9])
    p = MapParametrization(
        target="w_gain_e",
        expression="a + b * gaba",
        maps={"gaba": gaba},
        free_params={
            "a": FreeParam(init=0.0, bounds=(-1, 1)),
            "b": FreeParam(init=0.0, bounds=(-1, 1)),
        },
    )
    expected = 0.5 + 2.0 * gaba
    actual = p.evaluate({"a": 0.5, "b": 2.0})
    np.testing.assert_allclose(actual, expected)


def test_nonlinear_expression():
    """Confirm sympy handles standard math functions like exp."""
    myelin = np.array([1.0, 1.5, 2.0, 1.2, 0.8])
    p = MapParametrization(
        target="J",
        expression="a + b * exp(-c * myelin)",
        maps={"myelin": myelin},
        free_params={
            "a": FreeParam(0.0, (-1, 5)),
            "b": FreeParam(1.0, (0, 10)),
            "c": FreeParam(0.5, (0, 5)),
        },
    )
    expected = 1.0 + 2.0 * np.exp(-0.5 * myelin)
    actual = p.evaluate({"a": 1.0, "b": 2.0, "c": 0.5})
    np.testing.assert_allclose(actual, expected)


def test_fixed_and_free_params():
    """Mix of fixed and free params (mimics a Stage-2 fit where some
    parameters were resolved in a prior stage)."""
    gaba = np.array([0.1, 0.5, 0.7, 0.3, 0.9])
    p = MapParametrization(
        target="w_gain_e",
        expression="a + b * gaba",
        maps={"gaba": gaba},
        free_params={"b": FreeParam(0.0, (-1, 1))},
        fixed_params={"a": 0.5},
    )
    expected = 0.5 + 1.5 * gaba
    actual = p.evaluate({"b": 1.5})
    np.testing.assert_allclose(actual, expected)


def test_typo_detection():
    """A symbol in the expression with no corresponding map/param raises."""
    gaba = np.array([0.1, 0.5, 0.7])
    with pytest.raises(ValueError, match="GABA"):
        MapParametrization(
            target="x",
            expression="a + b * GABA",   # uppercase
            maps={"gaba": gaba},          # lowercase
            free_params={
                "a": FreeParam(0, (-1, 1)),
                "b": FreeParam(0, (-1, 1)),
            },
        )


def test_unused_symbol_detection():
    """A supplied map/param that isn't referenced raises (catches stale config)."""
    gaba = np.array([0.1, 0.5, 0.7])
    with pytest.raises(ValueError, match="not used"):
        MapParametrization(
            target="x",
            expression="a + b",  # gaba referenced nowhere
            maps={"gaba": gaba},
            free_params={
                "a": FreeParam(0, (-1, 1)),
                "b": FreeParam(0, (-1, 1)),
            },
        )


def test_scalar_only_parametrization():
    """No maps — pure scalar expression. Returns a 1-D array of length 1."""
    p = MapParametrization(
        target="something",
        expression="a + b",
        maps={},
        free_params={
            "a": FreeParam(0, (-1, 1)),
            "b": FreeParam(0, (-1, 1)),
        },
    )
    result = p.evaluate({"a": 2.0, "b": 3.0})
    np.testing.assert_allclose(result, np.array([5.0]))


def test_mismatched_map_lengths_raises():
    """All maps in one parametrization must share a node count."""
    with pytest.raises(ValueError, match="same length"):
        MapParametrization(
            target="x",
            expression="gaba + myelin",
            maps={
                "gaba": np.array([1.0, 2.0, 3.0]),
                "myelin": np.array([1.0, 2.0]),  # different length
            },
        )


def test_missing_free_value_raises():
    """Calling evaluate without supplying a required free param raises KeyError."""
    gaba = np.array([0.1, 0.5, 0.7])
    p = MapParametrization(
        target="x",
        expression="a + b * gaba",
        maps={"gaba": gaba},
        free_params={
            "a": FreeParam(0, (-1, 1)),
            "b": FreeParam(0, (-1, 1)),
        },
    )
    with pytest.raises(KeyError, match="b"):
        p.evaluate({"a": 0.5})


def test_extra_free_values_ignored():
    """Extra keys in free_values are tolerated (lets one theta dict serve
    multiple parametrizations)."""
    gaba = np.array([0.1, 0.5, 0.7])
    p = MapParametrization(
        target="x",
        expression="a + b * gaba",
        maps={"gaba": gaba},
        free_params={
            "a": FreeParam(0, (-1, 1)),
            "b": FreeParam(0, (-1, 1)),
        },
    )
    result = p.evaluate({"a": 0.5, "b": 1.0, "G": 99.0, "z": -7.0})
    np.testing.assert_allclose(result, 0.5 + gaba)


def test_compiled_function_is_cached():
    """The lambdified callable should only be built once."""
    gaba = np.array([0.1, 0.5, 0.7])
    p = MapParametrization(
        target="x",
        expression="a * gaba",
        maps={"gaba": gaba},
        free_params={"a": FreeParam(0, (-1, 1))},
    )
    p.evaluate({"a": 1.0})
    fn1 = p._compiled_fn
    p.evaluate({"a": 2.0})
    fn2 = p._compiled_fn
    assert fn1 is fn2  # same object reused