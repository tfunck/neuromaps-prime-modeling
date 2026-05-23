import numpy as np
from nmp_modeling.parametrization import MapParametrization
from nmp_modeling.adapters.neuronumba.fic import compute_j

def _get_model_class(model):
    """Return the Neuronumba model class from a model name or class."""
    if isinstance(model, str):
        if model == "GenericBEIDMF":
            from .models.generic_beidmf import GenericBEIDMF
            return GenericBEIDMF
        raise ValueError(f"Unknown Neuronumba model: {model}")
    return model

def _default_obs_var(model_class):
    """Return the default observable variable for supported models."""
    if model_class.__name__ == "GenericBEIDMF":
        return "re"
    return None

def _make_sigmas(sigma, n_state_vars):
    """Convert scalar or vector sigma to a vector matching state variables."""
    sigmas = np.asarray(sigma, dtype=float).reshape(-1)
    if sigmas.size == 1:
        return np.repeat(sigmas[0], n_state_vars)
    if sigmas.size != n_state_vars:
        raise ValueError(
            f"sigma must be scalar or length {n_state_vars}, got {sigmas.size}"
        )
    return sigmas


class NeuronumbaAdapter:
    def __init__(
        self,
        weights,
        parametrizations=None,
        fixed_model_attrs=None,
        model="GenericBEIDMF",
        g_param="G",
        dt=0.1,
        sigma=1e-2,
        sim_duration=440000.0,
        sim_warmup=0.0,
        sampling_period=1.0,
        tr=2000.0,
        obs_var=None,
        return_bold=True,
        fic_target_rate=None,
        fic_t_max=10000.0,
        fic_t_warmup=0.0,
        fic_max_trials=5000,
        fic_tolerance=0.005,
    ):
        from neuronumba.simulator.integrators.euler import EulerStochastic
        from neuronumba.simulator.simulator import simulate_nodelay
        from neuronumba.bold.stephan_2007 import BoldStephan2007
        from neuronumba.tools.random import set_seed

        self.weights = np.asarray(weights, dtype=float)
        if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError("weights must be a square matrix.")

        self.model_class = _get_model_class(model)
        self.fixed_model_attrs = dict(fixed_model_attrs or {})

        self.parametrizations = list(parametrizations or [])
        targets = [p.target for p in self.parametrizations]
        duplicates = sorted({x for x in targets if targets.count(x) > 1})
        if duplicates:
            raise ValueError(f"Duplicate parametrization target(s): {duplicates}")

        self.g_param = g_param
        self.dt = float(dt)
        self.sigma = sigma
        self.sim_duration = float(sim_duration)
        self.sim_warmup = float(sim_warmup)
        self.sampling_period = float(sampling_period)
        self.tr = float(tr)
        self.obs_var = obs_var or _default_obs_var(self.model_class)
        self.return_bold = bool(return_bold)

        if self.obs_var is None:
            raise ValueError("obs_var must be specified for this model.")

        self._EulerStochastic = EulerStochastic
        self._simulate_nodelay = simulate_nodelay
        self._BoldStephan2007 = BoldStephan2007
        self._set_seed = set_seed

        self.fic_target_rate = fic_target_rate
        self.fic_t_max = float(fic_t_max)
        self.fic_t_warmup = float(fic_t_warmup)
        self.fic_max_trials = int(fic_max_trials)
        self.fic_tolerance = float(fic_tolerance)

    def free_param_names(self):
        """Return theta parameters needed by this adapter."""
        names = {self.g_param}
        for p in self.parametrizations:
            names.update(p.free_params.keys())
        return sorted(names)

    def _model_attrs_from_theta(self, theta):
        """Build model attributes from fixed attributes and parametrizations."""
        if self.g_param not in theta:
            raise KeyError(f"Missing global coupling parameter: {self.g_param}")

        attrs = dict(self.fixed_model_attrs)
        attrs["g"] = float(theta[self.g_param])

        for p in self.parametrizations:
            missing = set(p.free_params) - set(theta)
            if missing:
                raise KeyError(
                    f"Missing parameter(s) for {p.target}: {sorted(missing)}"
                )

            free_values = {k: theta[k] for k in p.free_params}
            value = np.asarray(p.evaluate(free_values), dtype=float).reshape(-1)

            if value.size not in (1, self.weights.shape[0]):
                raise ValueError(
                    f"{p.target} has length {value.size}; expected 1 or "
                    f"{self.weights.shape[0]}"
                )

            attrs[p.target] = value

        attrs.update(dict(theta.get("_model_attrs", {}) or {}))
        if attrs.get("auto_fic", False) and "J" in attrs:
            raise ValueError(
                "auto_fic=True but J is also provided. "
                "Use auto_fic=True without J, or provide J with auto_fic=False."
            )

        return attrs

    def _maybe_compute_missing_j(self, attrs, integrator, seed):
        """Compute J when auto_fic is False and no J is provided."""
        model = self.model_class()
        if not hasattr(model, "auto_fic"):
            return attrs
        if attrs.get("auto_fic", getattr(model, "auto_fic")):
            return attrs
        if "J" in attrs:
            return attrs

        model.set_attributes(attrs)
        attrs["J"] = compute_j(
            model=model,
            weights=self.weights,
            g=attrs["g"],
            integrator=integrator,
            seed=seed,
            target_rate=self.fic_target_rate,
            t_max=self.fic_t_max,
            t_warmup=self.fic_t_warmup,
            max_trials=self.fic_max_trials,
            tolerance=self.fic_tolerance,
        )
        return attrs

    def _make_integrator(self, model):
        """Create an Euler stochastic integrator for the current model."""
        sigmas = _make_sigmas(self.sigma, model.n_state_vars)
        return self._EulerStochastic(dt=self.dt, sigmas=sigmas)

    def prepare_theta(self, theta, seed):
        """Prepare theta once for one run seed."""
        attrs = self._model_attrs_from_theta(theta)
        model = self.model_class()
        integrator = self._make_integrator(model)
        attrs = self._maybe_compute_missing_j(attrs, integrator, seed)

        if "J" not in attrs:
            return theta

        out = dict(theta)
        model_attrs = dict(out.get("_model_attrs", {}) or {})
        model_attrs["J"] = attrs["J"]
        out["_model_attrs"] = model_attrs
        return out

    def simulate(self, theta, seed):
        """Run one Neuronumba simulation."""
        model = self.model_class()
        integrator = self._make_integrator(model)
        attrs = self._model_attrs_from_theta(theta)
        attrs = self._maybe_compute_missing_j(attrs, integrator, seed)
        model.set_attributes(attrs)

        self._set_seed(int(seed))
        signal = self._simulate_nodelay(
            model,
            integrator,
            self.weights,
            self.obs_var,
            self.sampling_period,
            self.sim_duration,
            self.sim_warmup,
        )

        if not self.return_bold:
            return signal

        bold_model = self._BoldStephan2007().configure()
        bold_model.tr = self.tr
        return bold_model.compute_bold(signal, self.sampling_period)
