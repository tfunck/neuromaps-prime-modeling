"""nmp_modeling: molecular priors for whole-brain modeling.

Couples brain maps (e.g., PET receptor densities from neuromaps and
Neuromaps-PRIME) to whole-brain simulators, and fits the resulting models
to empirical fMRI observables.
"""

__version__ = "0.1.0"

from .parametrization import FreeParam, MapParametrization, WeightedMapParametrization
from .observable_specs import (
    EmpiricalTarget,
    ObservableSpec,
    OBSERVABLE_SPECS,
    get_observable_spec,
    make_empirical_target,
    make_empirical_targets,
)
from .fitting import (
    ContinuousFitResult,
    EvaluationResult,
    SweepResult,
    continuous_fit,
    evaluate_theta,
    grid_sweep,
)
from .optimization import OptimizationResult, optimize
from .io import LoadedData, load_data
from .preprocessing import make_bandpass_filter
from . import observables, objectives, hopf

__all__ = [
    "__version__",
    "FreeParam", "MapParametrization", "WeightedMapParametrization",
    "EmpiricalTarget", "ObservableSpec", "OBSERVABLE_SPECS",
    "get_observable_spec", "make_empirical_target", "make_empirical_targets",
    "ContinuousFitResult", "EvaluationResult", "SweepResult",
    "continuous_fit", "evaluate_theta", "grid_sweep",
    "OptimizationResult", "optimize",
    "LoadedData", "load_data",
    "make_bandpass_filter",
    "observables", "objectives", "hopf",
]
