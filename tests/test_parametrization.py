import numpy as np
import pytest
from nmp_modeling.parametrization import FreeParam, MapParametrization

def test_free_param_validates_and_normalizes_values():
    # FreeParam should convert numeric inputs to floats.
    p = FreeParam(init=0, bounds=(-1, 1))
    assert p.init == 0.0
    assert p.bounds == (-1.0, 1.0)

def test_free_param_rejects_invalid_bounds_length():
    # Bounds must have exactly two values.
    with pytest.raises(ValueError, match="bounds must contain exactly two values"):
        FreeParam(init=0.0, bounds=(0.0,))

def test_free_param_rejects_reversed_bounds():
    # Lower bound must be smaller than upper bound.
    with pytest.raises(ValueError, match="lower < upper"):
        FreeParam(init=0.0, bounds=(1.0, -1.0))

def test_free_param_rejects_init_outside_bounds():
    # Initial value must lie inside the parameter bounds.
    with pytest.raises(ValueError, match="init must lie within bounds"):
        FreeParam(init=2.0, bounds=(-1.0, 1.0))

def test_linear_expression_with_map_and_free_params():
    # A simple linear expression should be evaluated node-wise.
    gaba = np.array([0.1, 0.2, 0.3])
    p = MapParametrization(
        target="M_e",
        expression="a + b * gaba",
        maps={"gaba": gaba},
        free_params={
            "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            "b": FreeParam(init=1.0, bounds=(0.0, 3.0)),
        },
    )
    result = p.evaluate({"a": 0.5, "b": 2.0})
    expected = 0.5 + 2.0 * gaba
    np.testing.assert_allclose(result, expected)

def test_nonlinear_expression_with_map():
    # SymPy expressions should support NumPy-compatible nonlinear functions.
    myelin = np.array([0.0, 1.0, 2.0])
    p = MapParametrization(
        target="w_ee",
        expression="a + b * exp(-c * myelin)",
        maps={"myelin": myelin},
        free_params={
            "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            "b": FreeParam(init=1.0, bounds=(0.0, 3.0)),
            "c": FreeParam(init=1.0, bounds=(0.0, 3.0)),
        },
    )
    result = p.evaluate({"a": 0.5, "b": 2.0, "c": 1.5})
    expected = 0.5 + 2.0 * np.exp(-1.5 * myelin)
    np.testing.assert_allclose(result, expected)

def test_expression_with_fixed_and_free_params():
    # Fixed parameters should be inserted automatically during evaluation.
    receptor = np.array([1.0, 2.0, 3.0])
    p = MapParametrization(
        target="M_i",
        expression="base + alpha * receptor",
        maps={"receptor": receptor},
        fixed_params={"base": 1.0},
        free_params={
            "alpha": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
        },
    )
    result = p.evaluate({"alpha": 0.25})
    expected = 1.0 + 0.25 * receptor
    np.testing.assert_allclose(result, expected)

def test_scalar_expression_without_maps():
    # MapParametrization can also represent a scalar expression.
    p = MapParametrization(
        target="gain_e",
        expression="a + b",
        free_params={
            "a": FreeParam(init=1.0, bounds=(0.0, 2.0)),
            "b": FreeParam(init=2.0, bounds=(0.0, 3.0)),
        },
    )
    result = p.evaluate({"a": 1.5, "b": 2.5})
    np.testing.assert_allclose(result, np.array([4.0]))

def test_constant_expression_without_maps():
    # A constant expression should return a length-one array.
    p = MapParametrization(
        target="M_e",
        expression="1.0",
    )
    result = p.evaluate({})
    np.testing.assert_allclose(result, np.array([1.0]))

def test_missing_symbol_in_expression_raises():
    # Every symbol used in the expression must be supplied.
    gaba = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="not supplied"):
        MapParametrization(
            target="M_e",
            expression="a + b * gaba",
            maps={"gaba": gaba},
            free_params={
                "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            },
        )

def test_extra_supplied_symbol_raises():
    # Supplied maps or parameters must actually appear in the expression.
    gaba = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="not used in expression"):
        MapParametrization(
            target="M_e",
            expression="a",
            maps={"gaba": gaba},
            free_params={
                "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            },
        )

def test_duplicate_symbol_between_maps_and_free_params_raises():
    # The same symbol cannot be both a map and a free parameter.
    with pytest.raises(ValueError, match="both maps and free_params"):
        MapParametrization(
            target="M_e",
            expression="a",
            maps={"a": np.array([1.0, 2.0])},
            free_params={
                "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            },
        )

def test_duplicate_symbol_between_maps_and_fixed_params_raises():
    # The same symbol cannot be both a map and a fixed parameter.
    with pytest.raises(ValueError, match="both maps and fixed_params"):
        MapParametrization(
            target="M_e",
            expression="a",
            maps={"a": np.array([1.0, 2.0])},
            fixed_params={"a": 1.0},
        )

def test_duplicate_symbol_between_free_and_fixed_params_raises():
    # The same symbol cannot be both a free and fixed parameter.
    with pytest.raises(ValueError, match="both free_params and fixed_params"):
        MapParametrization(
            target="M_e",
            expression="a",
            free_params={
                "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            },
            fixed_params={"a": 1.0},
        )

def test_invalid_target_raises():
    # Target must be a non-empty string.
    with pytest.raises(ValueError, match="target must be a non-empty string"):
        MapParametrization(
            target="",
            expression="1.0",
        )

def test_invalid_expression_raises():
    # Expression must be a non-empty string.
    with pytest.raises(ValueError, match="expression must be a non-empty string"):
        MapParametrization(
            target="M_e",
            expression="",
        )

def test_invalid_symbol_name_raises():
    # Symbol names in maps, free_params, and fixed_params must be non-empty strings.
    with pytest.raises(ValueError, match="invalid symbol name"):
        MapParametrization(
            target="M_e",
            expression="a",
            maps={"": np.array([1.0, 2.0])},
        )

def test_maps_are_flattened_before_length_check():
    # Row-vector maps should be flattened before checking length.
    receptor = np.array([[1.0, 2.0, 3.0]])
    p = MapParametrization(
        target="gain_map_e",
        expression="receptor",
        maps={"receptor": receptor},
    )
    result = p.evaluate({})
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))

def test_empty_map_raises():
    # Empty maps are not valid.
    with pytest.raises(ValueError, match="is empty"):
        MapParametrization(
            target="gain_map_e",
            expression="receptor",
            maps={"receptor": np.array([])},
        )

def test_maps_with_different_lengths_raise():
    # All maps in the same expression must have the same length.
    with pytest.raises(ValueError, match="same length"):
        MapParametrization(
            target="M_e",
            expression="a + receptor + myelin",
            maps={
                "receptor": np.array([1.0, 2.0, 3.0]),
                "myelin": np.array([1.0, 2.0]),
            },
            free_params={
                "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            },
        )

def test_fixed_param_must_be_scalar():
    # Fixed parameters should be scalar values, not node-wise arrays.
    with pytest.raises(ValueError, match="must be scalar"):
        MapParametrization(
            target="M_e",
            expression="base + receptor",
            maps={"receptor": np.array([1.0, 2.0, 3.0])},
            fixed_params={"base": np.array([1.0, 2.0, 3.0])},
        )

def test_free_param_must_be_free_param_object():
    # free_params values must be FreeParam objects.
    with pytest.raises(TypeError, match="must be a FreeParam object"):
        MapParametrization(
            target="M_e",
            expression="alpha",
            free_params={"alpha": 0.0},
        )

def test_evaluate_requires_all_free_values():
    # Evaluation should fail if a needed free value is missing.
    p = MapParametrization(
        target="M_e",
        expression="a + b",
        free_params={
            "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
            "b": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
        },
    )
    with pytest.raises(KeyError, match="Missing free parameter"):
        p.evaluate({"a": 0.5})

def test_evaluate_ignores_extra_free_values():
    # Extra values in theta-like dictionaries should be ignored.
    p = MapParametrization(
        target="M_e",
        expression="a",
        free_params={
            "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
        },
    )
    result = p.evaluate({"a": 0.5, "G": 2.0, "unused": 99.0})
    np.testing.assert_allclose(result, np.array([0.5]))

def test_evaluate_rejects_vector_free_value():
    # Free parameter values should be scalar at evaluation time.
    p = MapParametrization(
        target="M_e",
        expression="a",
        free_params={
            "a": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
        },
    )
    with pytest.raises(ValueError, match="must be scalar"):
        p.evaluate({"a": np.array([0.1, 0.2])})

def test_compiled_function_is_cached():
    # The lambdified function should be compiled once and reused.
    receptor = np.array([1.0, 2.0, 3.0])
    p = MapParametrization(
        target="M_e",
        expression="1 + alpha * receptor",
        maps={"receptor": receptor},
        free_params={
            "alpha": FreeParam(init=0.0, bounds=(-1.0, 1.0)),
        },
    )
    assert p._compiled_fn is None
    p.evaluate({"alpha": 0.5})
    first_compiled = p._compiled_fn
    p.evaluate({"alpha": 0.25})
    second_compiled = p._compiled_fn
    assert first_compiled is second_compiled
