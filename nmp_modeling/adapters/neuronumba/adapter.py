import numpy as np
from nmp_modeling.parametrization import MapParametrization
from nmp_modeling.adapters.neuronumba.fic import compute_j
from nmp_modeling.adapters.neuronumba.integrators import EulerStochastic

_NOISE_TARGETS = {"sigma", "sigmas"}

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

def _noise_template(model):
    """Return the model noise template as a 1D float array."""
    if hasattr(model, "get_noise_template"):
        template = np.asarray(model.get_noise_template(), dtype=float).reshape(-1)
    else:
        template = np.ones(model.n_state_vars, dtype=float)

    if template.size != model.n_state_vars:
        raise ValueError(
            "Noise template length must match the number of state variables."
        )
    return template

def _make_sigmas(sigma, model, n_rois):
    """Convert user sigma input to full state-by-ROI-compatible sigmas."""
    template = _noise_template(model)
    active = np.flatnonzero(template != 0.0)
    n_state_vars = model.n_state_vars
    n_active = active.size

    if n_active == 0:
        return np.zeros(n_state_vars, dtype=float)

    arr = np.asarray(sigma, dtype=float)

    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0]) * template

    if arr.ndim == 1:
        values = arr.reshape(-1)

        state_matches = []
        if values.size == n_active:
            state_matches.append(f"n_active={n_active}")
        if values.size == n_state_vars:
            state_matches.append(f"n_state_vars={n_state_vars}")
        if values.size == n_rois and state_matches:
            raise ValueError(
                f"Ambiguous 1D sigma length {values.size}: it matches "
                f"n_rois={n_rois} and {', '.join(state_matches)}. "
                "Pass a 2D array with shape (n_active, n_rois) or "
                "(n_state_vars, n_rois) to specify ROI-specific sigmas."
            )

        if values.size == n_active:
            out = np.zeros(n_state_vars, dtype=float)
            out[active] = values
            return out
        if values.size == n_state_vars:
            return values * (template != 0.0)
        if values.size == n_rois:
            out = np.zeros((n_state_vars, n_rois), dtype=float)
            out[active, :] = values[None, :]
            return out
        if values.size == n_active * n_rois:
            active_matrix = values.reshape(n_active, n_rois)
            out = np.zeros((n_state_vars, n_rois), dtype=float)
            out[active, :] = active_matrix
            return out
        if values.size == n_state_vars * n_rois:
            out = values.reshape(n_state_vars, n_rois)
            out = out * (template != 0.0)[:, None]
            return out

    if arr.ndim == 2:
        if arr.shape == (n_active, n_rois):
            out = np.zeros((n_state_vars, n_rois), dtype=float)
            out[active, :] = arr
            return out
        if arr.shape == (n_state_vars, n_rois):
            return arr * (template != 0.0)[:, None]

    raise ValueError(
        "sigma must be scalar, length n_active, length n_state_vars, "
        "length n_rois, length n_active*n_rois, length n_state_vars*n_rois, "
        "shape (n_active, n_rois), or shape (n_state_vars, n_rois)."
    )


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

    def _resolve_auto_fic(self, attrs):
        """Return the effective auto_fic setting for the current model attrs."""
        model = self.model_class()
        if not hasattr(model, "auto_fic"):
            return None
        if "auto_fic" in attrs:
            return bool(attrs["auto_fic"])
        return bool(getattr(model, "auto_fic"))

    def _model_attrs_from_theta(self, theta):
        """Build model attributes and optional sigma values from theta."""
        if self.g_param not in theta:
            raise KeyError(f"Missing global coupling parameter: {self.g_param}")

        attrs = dict(self.fixed_model_attrs)
        attrs["g"] = float(theta[self.g_param])

        sigma_value = None
        if "sigmas" in theta:
            sigma_value = theta["sigmas"]
        elif "sigma" in theta:
            sigma_value = theta["sigma"]

        for p in self.parametrizations:
            missing = set(p.free_params) - set(theta)
            if missing:
                raise KeyError(
                    f"Missing parameter(s) for {p.target}: {sorted(missing)}"
                )

            free_values = {k: theta[k] for k in p.free_params}
            raw_value = np.asarray(p.evaluate(free_values), dtype=float)
            if p.target in _NOISE_TARGETS:
                if sigma_value is not None:
                    raise ValueError(
                        "Noise was provided both directly in theta and by "
                        "a MapParametrization target."
                    )
                sigma_value = raw_value
                continue
            value = raw_value.reshape(-1)

            if value.size not in (1, self.weights.shape[0]):
                raise ValueError(
                    f"{p.target} has length {value.size}; expected 1 or "
                    f"{self.weights.shape[0]}"
                )

            attrs[p.target] = float(value[0]) if value.size == 1 else value

        attrs.update(dict(theta.get("_model_attrs", {}) or {}))
        auto_fic = self._resolve_auto_fic(attrs)
        if auto_fic is True and "J" in attrs:
            raise ValueError(
                "auto_fic=True but J is also provided. "
                "Use auto_fic=True without J, or provide J with auto_fic=False."
            )

        return attrs, sigma_value

    def _maybe_compute_missing_j(self, attrs, integrator, seed):
        """Compute J when auto_fic is False and no J is provided."""
        auto_fic = self._resolve_auto_fic(attrs)
        if auto_fic is None or auto_fic:
            return attrs
        if "J" in attrs:
            return attrs

        model = self.model_class()
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

    def _make_integrator(self, model, sigma_value=None):
        """Create an Euler stochastic integrator for the current model."""
        sigma = self.sigma if sigma_value is None else sigma_value
        sigmas = _make_sigmas(sigma, model, self.weights.shape[0])
        return self._EulerStochastic(dt=self.dt, sigmas=sigmas)

    def prepare_theta(self, theta, seed):
        """Prepare theta once for one run seed."""
        attrs, sigma_value = self._model_attrs_from_theta(theta)
        model = self.model_class()
        integrator = self._make_integrator(model, sigma_value)
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
        attrs, sigma_value = self._model_attrs_from_theta(theta)
        model = self.model_class()
        integrator = self._make_integrator(model, sigma_value)
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
