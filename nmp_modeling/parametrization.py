from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sympy as sp


@dataclass
class FreeParam:
    init: float
    bounds: Tuple[float, float]

    def __post_init__(self):
        """Validate and normalize a scalar free parameter."""
        if len(self.bounds) != 2:
            raise ValueError("bounds must contain exactly two values: (lower, upper).")
        lower, upper = float(self.bounds[0]), float(self.bounds[1])
        init = float(self.init)
        if lower >= upper:
            raise ValueError("bounds must satisfy lower < upper.")
        if not (lower <= init <= upper):
            raise ValueError("init must lie within bounds.")
        self.init = init
        self.bounds = (lower, upper)


@dataclass
class MapParametrization:
    target: str
    expression: str
    maps: Dict[str, np.ndarray] = field(default_factory=dict)
    free_params: Dict[str, FreeParam] = field(default_factory=dict)
    fixed_params: Dict[str, float] = field(default_factory=dict)

    _expr: sp.Expr = field(init=False, repr=False)
    _symbol_order: Tuple[str, ...] = field(init=False, repr=False)
    _compiled_fn: Optional[Callable] = field(default=None, init=False, repr=False)
    _n_nodes: int = field(init=False, repr=False)

    def __post_init__(self):
        """Validate symbols, normalize maps, and prepare expression evaluation."""
        self._validate_basic_fields()

        self.maps = dict(self.maps or {})
        self.free_params = dict(self.free_params or {})
        self.fixed_params = dict(self.fixed_params or {})

        self._validate_symbol_names()
        self._validate_no_duplicate_symbols()

        self._expr = sp.sympify(self.expression)
        symbols_in_expr = {s.name for s in self._expr.free_symbols}

        supplied_symbols = (
            set(self.maps)
            | set(self.free_params)
            | set(self.fixed_params)
        )
        missing = symbols_in_expr - supplied_symbols
        extra = supplied_symbols - symbols_in_expr
        if missing:
            raise ValueError(
                f"Expression uses symbol(s) not supplied by maps, free_params, "
                f"or fixed_params: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"Supplied symbol(s) not used in expression: {sorted(extra)}"
            )

        self.maps = self._normalize_maps(self.maps)
        self.fixed_params = self._normalize_fixed_params(self.fixed_params)
        self._validate_free_params(self.free_params)

        self._symbol_order = tuple(
            sorted(self.maps)
            + sorted(self.fixed_params)
            + sorted(self.free_params)
        )

    def _validate_basic_fields(self):
        """Check target and expression fields."""
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty string.")
        if not isinstance(self.expression, str) or not self.expression:
            raise ValueError("expression must be a non-empty string.")

    def _validate_symbol_names(self):
        """Check that all supplied symbol names are strings."""
        for group_name, group in (
            ("maps", self.maps),
            ("free_params", self.free_params),
            ("fixed_params", self.fixed_params),
        ):
            bad_names = [k for k in group if not isinstance(k, str) or not k]
            if bad_names:
                raise ValueError(
                    f"{group_name} contains invalid symbol name(s): {bad_names}"
                )

    def _validate_no_duplicate_symbols(self):
        """Prevent the same symbol from appearing in multiple input groups."""
        groups = {
            "maps": set(self.maps),
            "free_params": set(self.free_params),
            "fixed_params": set(self.fixed_params),
        }
        pairs = (
            ("maps", "free_params"),
            ("maps", "fixed_params"),
            ("free_params", "fixed_params"),
        )
        for left, right in pairs:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(
                    f"Symbol(s) cannot appear in both {left} and {right}: "
                    f"{sorted(overlap)}"
                )

    def _normalize_maps(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert maps to one-dimensional float arrays and check lengths."""
        normalized = {}
        n_nodes = None

        for name, values in maps.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 0:
                raise ValueError(f"Map '{name}' is empty.")
            if n_nodes is None:
                n_nodes = arr.size
            elif arr.size != n_nodes:
                raise ValueError(
                    f"All maps must have the same length. "
                    f"Map '{name}' has length {arr.size}, expected {n_nodes}."
                )
            normalized[name] = arr

        self._n_nodes = int(n_nodes) if n_nodes is not None else 1
        return normalized

    def _normalize_fixed_params(self, fixed_params: Dict[str, float]) -> Dict[str, float]:
        """Convert fixed parameters to floats."""
        normalized = {}

        for name, value in fixed_params.items():
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size != 1:
                raise ValueError(f"Fixed parameter '{name}' must be scalar.")
            normalized[name] = float(arr[0])

        return normalized

    def _validate_free_params(self, free_params: Dict[str, FreeParam]):
        """Check that all free parameters are FreeParam objects."""
        for name, value in free_params.items():
            if not isinstance(value, FreeParam):
                raise TypeError(
                    f"Free parameter '{name}' must be a FreeParam object."
                )

    def _compile(self) -> Callable:
        """Compile the symbolic expression into a NumPy-compatible function."""
        if self._compiled_fn is None:
            symbols = [sp.Symbol(name) for name in self._symbol_order]
            self._compiled_fn = sp.lambdify(symbols, self._expr, modules="numpy")
        return self._compiled_fn

    def evaluate(self, free_values: Dict[str, float]) -> np.ndarray:
        """Evaluate the expression using maps, fixed parameters, and free values."""
        free_values = dict(free_values or {})
        missing = set(self.free_params) - set(free_values)
        if missing:
            raise KeyError(f"Missing free parameter value(s): {sorted(missing)}")

        fn = self._compile()
        args = []

        for name in self._symbol_order:
            if name in self.maps:
                args.append(self.maps[name])
            elif name in self.fixed_params:
                args.append(self.fixed_params[name])
            else:
                arr = np.asarray(free_values[name], dtype=float).reshape(-1)
                if arr.size != 1:
                    raise ValueError(f"Free parameter '{name}' must be scalar.")
                args.append(float(arr[0]))

        result = np.asarray(fn(*args), dtype=float).reshape(-1)
        if result.size not in (1, self._n_nodes):
            raise ValueError(
                f"Expression for target '{self.target}' returned length "
                f"{result.size}, expected 1 or {self._n_nodes}."
            )
        return result


@dataclass
class WeightedMapParametrization:
    target: str
    maps: Dict[str, np.ndarray]
    weight_params: Union[FreeParam, Dict[str, FreeParam]]
    weight_prefix: str = "w_"

    free_params: Dict[str, FreeParam] = field(init=False)
    _map_names: Tuple[str, ...] = field(init=False, repr=False)
    _map_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Validate maps and prepare free weight parameters."""
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty string.")

        self.maps = dict(self.maps or {})
        if not self.maps:
            raise ValueError("At least one map is required.")

        if not isinstance(self.weight_prefix, str):
            raise TypeError("weight_prefix must be a string.")

        normalized = {}
        n_nodes = None

        for name, values in self.maps.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid map name: {name!r}.")

            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 0:
                raise ValueError(f"Map '{name}' is empty.")

            if n_nodes is None:
                n_nodes = arr.size
            elif arr.size != n_nodes:
                raise ValueError(
                    f"All maps must have the same length. "
                    f"Map '{name}' has length {arr.size}, expected {n_nodes}."
                )

            normalized[name] = arr

        self.maps = normalized
        self._map_names = tuple(self.maps)

        if isinstance(self.weight_params, FreeParam):
            specs = {name: self.weight_params for name in self._map_names}
        else:
            specs = dict(self.weight_params or {})
            missing = set(self.maps) - set(specs)
            extra = set(specs) - set(self.maps)

            if missing:
                raise ValueError(f"Missing weight parameter(s) for map(s): {sorted(missing)}")
            if extra:
                raise ValueError(f"Weight parameter(s) supplied for unknown map(s): {sorted(extra)}")

            for name, spec in specs.items():
                if not isinstance(spec, FreeParam):
                    raise TypeError(f"Weight parameter for map '{name}' must be a FreeParam object.")

        self.free_params = {
            f"{self.weight_prefix}{name}": FreeParam(specs[name].init, specs[name].bounds)
            for name in self._map_names
        }

        self._map_matrix = np.stack(
            [self.maps[name] for name in self._map_names],
            axis=0,
        )

    def evaluate(self, free_values: Dict[str, float]) -> np.ndarray:
        """Return the weighted sum of all maps."""
        free_values = dict(free_values or {})
        missing = set(self.free_params) - set(free_values)
        if missing:
            raise KeyError(f"Missing free parameter value(s): {sorted(missing)}")

        weights = []

        for name in self._map_names:
            param_name = f"{self.weight_prefix}{name}"
            arr = np.asarray(free_values[param_name], dtype=float).reshape(-1)

            if arr.size != 1:
                raise ValueError(f"Free parameter '{param_name}' must be scalar.")

            weights.append(float(arr[0]))

        return np.asarray(weights, dtype=float) @ self._map_matrix
