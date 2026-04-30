"""Layer 1: simulator-agnostic parametrization of brain maps.

A ``MapParametrization`` ties together (a) a target variable in some
model's equations, (b) a string expression like ``"a + b * gaba"``,
(c) the brain maps referenced in that expression, and (d) a set of
*free* and *fixed* scalar parameters. It knows how to evaluate itself
at given parameter values to produce a per-node numpy vector.

This module contains no simulator imports and could equally well feed
a Kuramoto adapter, a REACT integration, or any other downstream tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp


@dataclass
class FreeParam:
    """Specification for a single free scalar parameter.

    Carries an initial value and bounds. Bounds are required because
    every reasonable optimiser (grid search, differential evolution,
    CMA-ES) needs them, and asking for them now costs nothing.

    Attributes
    ----------
    init : float
        Initial value. Used as the centre point for grid construction
        if no explicit grid is supplied, and as the starting guess for
        local optimisers in future versions.
    bounds : tuple of (float, float)
        (lower, upper) inclusive bounds.
    """
    init: float
    bounds: Tuple[float, float]


@dataclass
class MapParametrization:
    """Ties a target model variable to an expression over brain maps + scalars.

    A MapParametrization says: "the model attribute named ``target`` should
    be set to the per-node array obtained by evaluating ``expression``,
    where the symbols in the expression are either keys of ``maps`` (which
    resolve to numpy arrays) or names in ``free_params``/``fixed_params``
    (which resolve to scalars)."

    Parameters
    ----------
    target : str
        Name of the model attribute this parametrization controls. For
        Neuronumba's Deco2018 this might be ``"w_gain_e"``, ``"J"``, etc.
        The adapter is responsible for knowing whether the named attribute
        exists on the model class — this dataclass does not validate.
    expression : str
        A SymPy-parseable expression. Allowed: arithmetic (``+ - * /``),
        powers (``**``), standard functions (``exp``, ``log``, ``sqrt``,
        trig). Disallowed: anything sympify rejects. The parser is safe;
        it does not execute arbitrary Python.
    maps : dict[str, np.ndarray]
        Mapping from symbol name (as it appears in ``expression``) to a
        per-node numpy array. All arrays must have the same length, which
        becomes the number of nodes for this parametrization. The length
        is checked at construction time.
    free_params : dict[str, FreeParam]
        Scalar parameters that will be varied during fitting. Keys must
        appear as symbols in ``expression``.
    fixed_params : dict[str, float]
        Scalar parameters that are held fixed at known values. Useful for
        carrying forward a value from a previous fitting stage. Keys must
        appear as symbols in ``expression``.

    Notes
    -----
    The set ``free_params.keys() | fixed_params.keys() | maps.keys()`` must
    exactly match the free symbols of ``expression``. We check this in
    ``__post_init__``. If the user references a symbol that isn't supplied,
    or supplies one that isn't referenced, you get a clear error up-front
    rather than a confusing failure inside a simulation loop.

    The compiled numpy callable is cached on first use (``_compiled_fn``).
    """
    target: str
    expression: str
    maps: Dict[str, np.ndarray]
    free_params: Dict[str, FreeParam] = field(default_factory=dict)
    fixed_params: Dict[str, float] = field(default_factory=dict)

    # Set in __post_init__; not user-facing.
    _expr: sp.Expr = field(init=False, repr=False)
    _symbol_order: List[str] = field(init=False, repr=False)
    _compiled_fn: Optional[Callable] = field(default=None, init=False, repr=False)
    _n_nodes: int = field(init=False, repr=False)

    def __post_init__(self):
        # Parse the expression. sympify is safe — it does not exec.
        self._expr = sp.sympify(self.expression)

        # Names actually used in the expression.
        symbols_in_expr = {s.name for s in self._expr.free_symbols}

        # Names the user supplied across all three sources.
        supplied = (
            set(self.maps.keys())
            | set(self.free_params.keys())
            | set(self.fixed_params.keys())
        )

        # Catch typos early. This avoids the common failure mode where the
        # user writes "a + b * gaba" but has the array under the key "GABA".
        missing = symbols_in_expr - supplied
        extra = supplied - symbols_in_expr
        if missing:
            raise ValueError(
                f"Expression {self.expression!r} references symbols "
                f"{sorted(missing)} that are not in maps, free_params, "
                f"or fixed_params."
            )
        if extra:
            raise ValueError(
                f"maps/free_params/fixed_params include symbols {sorted(extra)} "
                f"that are not used in expression {self.expression!r}."
            )

        # Check map shapes. All maps for one parametrization must have the
        # same number of nodes.
        if self.maps:
            shapes = {k: np.asarray(v).shape for k, v in self.maps.items()}
            lengths = {k: s[0] if len(s) >= 1 else 1 for k, s in shapes.items()}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) != 1:
                raise ValueError(
                    f"All maps in a parametrization must have the same length; "
                    f"got {lengths}."
                )
            self._n_nodes = unique_lengths.pop()
        else:
            # Pure scalar parametrization — no maps. Allowed (e.g. a
            # constant offset). Number of nodes is 1; broadcasting handles
            # the rest at the adapter layer.
            self._n_nodes = 1

        # Coerce all map arrays to 1-D float numpy.
        self.maps = {
            k: np.asarray(v, dtype=float).reshape(-1)
            for k, v in self.maps.items()
        }

        # Lock down the symbol order for lambdify. This MUST be deterministic
        # so positional calls into the compiled function stay correct.
        self._symbol_order = (
            sorted(self.maps.keys())
            + sorted(self.fixed_params.keys())
            + sorted(self.free_params.keys())
        )

    def _compile(self) -> Callable:
        """Lazily build the numpy-vectorised callable from the SymPy expr."""
        if self._compiled_fn is None:
            symbols = [sp.Symbol(name) for name in self._symbol_order]
            # lambdify with "numpy" backend gives us elementwise broadcasting
            # over array arguments for free.
            self._compiled_fn = sp.lambdify(symbols, self._expr, modules="numpy")
        return self._compiled_fn

    def evaluate(self, free_values: Dict[str, float]) -> np.ndarray:
        """Evaluate the parametrization at given free-parameter values.

        Parameters
        ----------
        free_values : dict[str, float]
            Values for every key in ``self.free_params``. Extra keys are
            ignored (so you can pass a single flat ``theta`` dict that
            covers multiple parametrizations).

        Returns
        -------
        np.ndarray
            Per-node array of length ``self._n_nodes``.

        Raises
        ------
        KeyError
            If any free parameter is missing from ``free_values``.
        """
        fn = self._compile()

        # Assemble positional arguments in the locked symbol order.
        args = []
        for name in self._symbol_order:
            if name in self.maps:
                args.append(self.maps[name])
            elif name in self.fixed_params:
                args.append(self.fixed_params[name])
            else:
                # Must be a free param.
                if name not in free_values:
                    raise KeyError(
                        f"Free parameter {name!r} not provided in free_values; "
                        f"got keys {sorted(free_values.keys())}."
                    )
                args.append(free_values[name])

        result = fn(*args)

        # If the expression is purely scalar (no maps), lambdify returns a
        # scalar. Force a 1-D array so adapters see a consistent type.
        result = np.atleast_1d(np.asarray(result, dtype=float))
        return result