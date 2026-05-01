"""Layer 2: Neuronumba-specific translation adapter.

This is the ONLY module that imports Neuronumba. The fitting layer and
the parametrization layer are deliberately kept free of Neuronumba
imports so they remain reusable across backends and testable in
environments without Neuronumba installed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from nmp_modeling.parametrization import MapParametrization


class NeuronumbaAdapter:
    """Translation layer: NMP MapParametrizations -> Neuronumba simulation.

    Responsibilities
    ----------------
    1. Hold the model class reference (default Deco2014) and the SC matrix.
    2. Translate a flat ``theta`` dict into:
         - parametrization free-parameter values, evaluated to per-node arrays;
         - top-level model parameters (G, integrator settings, ...);
         - any fixed model attributes provided at construction.
    3. Run a single forward simulation and return BOLD.

    What it does NOT do
    -------------------
    - Multi-seed averaging (that's the fitting layer's job).
    - Empirical-data comparison (that's the fitting layer's job).
    - Parameter sweeping (that's the fitting layer's job).
    - FIC J caching (could be added here later if needed; deliberately
      omitted from v1 to keep the adapter readable).

    The single-shot ``simulate(theta, seed) -> bold`` interface is what
    the fitting layer treats as the "protocol" for any backend. Future
    Kuramoto, REACT, etc. adapters should expose the same method
    signature.

    Parameters
    ----------
    weights : np.ndarray
        Structural connectivity matrix, shape ``(n_nodes, n_nodes)``.
    parametrizations : list[MapParametrization]
        Each parametrization controls one model attribute (its ``target``).
        The adapter calls ``parametrization.evaluate(theta)`` and passes
        the resulting array via ``model.set_attributes({target: array})``.
    fixed_model_attrs : dict, optional
        Attributes set on every simulation, never varied. Examples:
        ``{"auto_fic": True}`` to let Neuronumba self-balance FIC.
    g_param : str
        Name under which the global coupling appears in ``theta``. Defaults
        to ``"G"``. Mapped onto ``model.configure(g=...)``.
    dt : float
        Integration step in ms. Default 0.1.
    sigma : float
        Noise amplitude for the Euler-stochastic integrator. Default 1e-2.
    sim_duration : float
        Total simulated time in ms. Default 440000 (the value used in the
        original notebook).
    sim_warmup : float
        Warmup time in ms before recording. Default 0.
    tr : float
        BOLD repetition time in ms. Default 2000.
    obs_var : str
        Which simulator state variable to feed into the BOLD model.
        Default ``"re"`` (excitatory firing rate) for Deco2014/2018.
    model_class : callable, optional
        Neuronumba model class. Defaults to importing ``Deco2014`` lazily.
        Pass your own to swap in a different model (e.g., a custom
        ``Deco2018`` variant from a separate module).
    """

    def __init__(
        self,
        weights: np.ndarray,
        parametrizations: Sequence[MapParametrization],
        fixed_model_attrs: Optional[Dict] = None,
        g_param: str = "G",
        dt: float = 0.1,
        sigma: float = 1e-2,
        sim_duration: float = 440000.0,
        sim_warmup: float = 0.0,
        tr: float = 2000.0,
        obs_var: str = "re",
        model_class: Optional[Callable] = None,
    ):
        # Lazy imports: keeps the module usable without Neuronumba for
        # testing the parametrization layer in isolation.
        from neuronumba.simulator.integrators.euler import EulerStochastic
        from neuronumba.simulator.simulator import simulate_nodelay
        from neuronumba.bold.stephan_2007 import BoldStephan2007
        from neuronumba.tools.random import set_seed

        self._EulerStochastic = EulerStochastic
        self._simulate_nodelay = simulate_nodelay
        self._BoldStephan2007 = BoldStephan2007
        self._set_seed = set_seed

        if model_class is None:
            from neuronumba.simulator.models.deco2014 import Deco2014
            model_class = Deco2014
        self._model_class = model_class

        self.weights = np.asarray(weights, dtype=float)
        self.parametrizations = list(parametrizations)
        self.fixed_model_attrs = dict(fixed_model_attrs or {})
        self.g_param = g_param
        self.dt = dt
        self.sigma = sigma
        self.sim_duration = sim_duration
        self.sim_warmup = sim_warmup
        self.tr = tr
        self.obs_var = obs_var

        # Cache of free-parameter names this adapter expects, derived from
        # the parametrizations. Used to validate ``theta`` dicts.
        self._param_free_names = set()
        for p in self.parametrizations:
            self._param_free_names |= set(p.free_params.keys())

    def free_param_names(self) -> List[str]:
        """All free names this adapter responds to: per-parametrization
        free params plus the global coupling parameter ``g_param``.
        """
        return sorted(self._param_free_names | {self.g_param})

    def simulate(self, theta: Dict[str, float], seed: int) -> np.ndarray:
        """Run one forward simulation and return BOLD.

        Parameters
        ----------
        theta : dict[str, float]
            Flat dict mapping free-parameter names to scalar values. Must
            contain at least every key in ``self.free_param_names()``.
            Extra keys are ignored (handy for compositional fitting where
            different parametrizations share the same theta dict).
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        np.ndarray
            BOLD signal, shape determined by Neuronumba's BoldStephan2007.
            Typically ``(n_timepoints, n_nodes)``.
        """
        self._set_seed(int(seed))

        # Build the model and apply fixed attrs first.
        model = self._model_class()
        attrs = dict(self.fixed_model_attrs)

        # Evaluate each parametrization at the current theta and stage its
        # output to be set on the model. We build the dict before calling
        # set_attributes so a single batched call is made.
        for p in self.parametrizations:
            free_vals = {k: theta[k] for k in p.free_params if k in theta}
            missing = set(p.free_params) - set(theta)
            if missing:
                raise KeyError(
                    f"theta is missing free params {sorted(missing)} required "
                    f"by parametrization with target {p.target!r}."
                )
            attrs[p.target] = p.evaluate(free_vals)

        model.set_attributes(attrs)

        # Configure with the SC matrix and global coupling. The g_param
        # name is what theta uses; Neuronumba's configure() takes ``g=``.
        if self.g_param not in theta:
            raise KeyError(
                f"theta missing global coupling parameter {self.g_param!r}."
            )
        model.configure(weights=self.weights, g=float(theta[self.g_param]))

        # Integrator: a 2-D state for Deco2014/2018 (excitatory + inhibitory)
        # so we pass a length-2 sigma vector.
        integrator = self._EulerStochastic(
            dt=self.dt,
            sigmas=np.array([self.sigma, self.sigma]),
        )

        # Simulate firing rates.
        signal = self._simulate_nodelay(
            model,
            integrator,
            self.weights,
            self.obs_var,
            1.0,                  # sampling period for the recorded signal
            self.sim_duration,
            self.sim_warmup,
        )

        # Convert firing rate to BOLD.
        bold_model = self._BoldStephan2007().configure()
        bold_model.tr = self.tr
        bold = bold_model.compute_bold(signal, 1.0)
        return bold