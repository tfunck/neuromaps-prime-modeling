# nmp_modeling

**Molecular priors for whole-brain modeling.** `nmp_modeling` couples spatial brain maps — for example PET-derived neurotransmitter receptor densities from [neuromaps](https://github.com/netneurolab/neuromaps) and [Neuromaps-PRIME](https://github.com/childmindresearch/neuromaps-prime) — to whole-brain simulators, and fits the resulting models to empirical fMRI observables.

You describe how a map enters a model with a symbolic expression such as `"a + b * gaba"`, where `a` and `b` are free parameters. The library parses the expression, translates it for the simulation backend, computes empirical and simulated observables with a shared registry, and fits the free parameters by grid sweep or continuous optimisation. Model simulation itself is delegated to existing packages through backend adapters.

## Status

Version 0.1.0 is the first tagged release. It is research software: the API is stable enough to cite, but expect it to evolve. The release reproduces three published receptor-informed modeling results (see [Examples and validation](#examples-and-validation)).

## What's included

| Component | Module | Notes |
|---|---|---|
| Map parametrisation | `parametrization` | `MapParametrization` (symbolic expression over one or more maps, SymPy-parsed, compiled to numpy) and `WeightedMapParametrization` (weighted sum over a map panel, e.g. 19 receptor maps) |
| Backend adapter | `adapters.neuronumba` | `NeuronumbaAdapter` wrapping [Neuronumba](https://github.com/neich/neuronumba) with generic Hopf, balanced excitation–inhibition DMF (BEI-DMF), and multiscale DMF models, plus feedback-inhibition control (FIC) |
| Observables and distances | `observables`, `objectives`, `observable_specs` | FC, GBC, sliding-window and phase FCD, lagged covariance and mutual information; Pearson/cosine, Frobenius, MSE, and Kolmogorov–Smirnov distances; a registry (`OBSERVABLE_SPECS`) pairing each observable with its empirical computation, aggregation, and distance |
| Fitting | `fitting`, `optimization` | `grid_sweep`, `continuous_fit`, and `optimize` (CMA-ES and particle-swarm); multi-seed, multi-subject evaluation; multi-target losses with weights; staged fitting by threading `result.params` into the next stage's `fixed` |
| Hopf utilities | `hopf.frequency`, `hopf.linear`, `hopf.gec` | Regional peak-frequency estimation; linearised Hopf covariance and stability checks; generative effective connectivity (GEC) fitting from lagged covariance in the cooperative–competitive formulation |
| Data I/O | `io`, `data`, `preprocessing` | Loading `.npz`/`.mat` inputs; subject/time/node shape inference; band-pass filtering |

## Install

```bash
git clone https://github.com/tfunck/neuromaps-prime-modeling.git
cd neuromaps-prime-modeling
pip install -e .
```

Optional extras:

```bash
pip install -e ".[neuronumba]"     # simulation backend (installs from GitHub)
pip install -e ".[optimization]"   # CMA-ES
pip install -e ".[io]"             # hdf5storage for MATLAB v7.3 files
pip install -e ".[all]"            # everything above plus pytest
```

Requires Python ≥ 3.9. Core dependencies are numpy, scipy, sympy, and numba.

## Quickstart

```python
import numpy as np
from nmp_modeling import (
    MapParametrization, FreeParam,
    make_empirical_target, grid_sweep, observables,
)
from nmp_modeling.adapters.neuronumba import NeuronumbaAdapter

# SC: (nodes, nodes) structural connectivity; bold: (subjects, time, nodes) empirical
# time series; receptor_map: (nodes,) e.g. 5-HT2A density from neuromaps.

# 1. Describe how the map enters the model: here it modulates excitatory gain
#    (M_e is the BEI-DMF excitatory gain attribute; default 1.0).
p_gain = MapParametrization(
    target="M_e",
    expression="a + b * receptor",
    maps={"receptor": receptor_map},
    free_params={
        "a": FreeParam(init=1.0, bounds=(0.8, 1.2)),
        "b": FreeParam(init=0.0, bounds=(-0.2, 0.5)),
    },
)

# 2. Build the backend adapter.
adapter = NeuronumbaAdapter(
    weights=SC,
    parametrizations=[p_gain],
    fixed_model_attrs={"auto_fic": True},
)

# 3. Define the empirical target from the observable registry.
target = make_empirical_target(bold, "fcd_ks", window_size=30, step=2)

# 4. Fit: sweep global coupling first, then the receptor gain with G fixed.
stage_a = grid_sweep(adapter, free_grid={"G": np.arange(0.5, 3.0, 0.25)},
                     fixed={"a": 1.0, "b": 0.0}, targets=target,
                     n_subjects=15, run_seeds=[11, 23, 37])
stage_b = grid_sweep(adapter, free_grid={"b": np.linspace(0, 0.4, 9)},
                     fixed={k: v for k, v in stage_a.params.items() if k != "b"},
                     targets=target, n_subjects=15, run_seeds=[11, 23, 37])
print(stage_b.params, stage_b.best_loss)
```

Available observable labels: `fc_corr`, `fc_corr_z`, `gbc_corr`, `gbc_corr_z`, `fc_frobenius`, `fc_mse`, `fc_mean_abs_diff`, `fc_distribution_ks`, `fcd_ks`, `phfcd_ks`. Several targets can be fit jointly by passing a list to `targets` with optional `target_weights`.

Parameter names in `free_grid`/`fixed` must match the names in `MapParametrization.free_params` exactly; the adapter evaluates each parametrisation from `theta` at simulation time, so a mismatch raises `KeyError` there.

## Examples and validation

The notebooks in `examples/` reproduce published analyses and are the validation record for this release. They require data that is not distributed with the package; each notebook states what it expects and where it comes from.

| Notebook | Reproduces | Model |
|---|---|---|
| `minimal_example.ipynb` | Global-coupling sweep on a 66-region connectome (Deco et al., 2014 setup) | BEI-DMF |
| `deco2018_two_stage_beidmf_fitting.ipynb` | Two-stage LSD fitting: coupling on placebo FCD, then 5-HT2A-modulated gain on LSD (Deco et al., *Curr Biol* 2018) | BEI-DMF |
| `nemo2026_group_neuromodulation_fitting.ipynb` | NEMO group analysis: 19 receptor maps jointly modulating regional Hopf noise across seven HCP task states (Deco et al., *Cell Rep* 2026) | Hopf + GEC, PSO |
| `luppi2026_hopf_gec_fitting.ipynb` | Cooperative–competitive GEC fitting from lagged covariance (Luppi et al., *Nat Neurosci* 2026) | Hopf + GEC |

## Layout

```
neuromaps-prime-modeling/
├── nmp_modeling/
│   ├── __init__.py            public API and __version__
│   ├── parametrization.py     MapParametrization, WeightedMapParametrization
│   ├── observables.py         FC, GBC, FCD, lagged covariance / MI
│   ├── objectives.py          distance functions
│   ├── observable_specs.py    observable registry, EmpiricalTarget
│   ├── fitting.py             grid_sweep, continuous_fit, evaluate_theta
│   ├── optimization.py        optimize (CMA-ES, PSO)
│   ├── data.py / io.py        input inference and loading
│   ├── preprocessing.py       band-pass filter
│   ├── hopf/                  frequency, linear, gec
│   └── adapters/neuronumba/   adapter, integrators, FIC, generic models
├── examples/                  validation notebooks (see above)
├── tests/                     pytest suite (mock adapter; Neuronumba not required)
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite covers parametrisation, observables and distances, and the fitting layer using a mock adapter. The Neuronumba adapter is exercised by the example notebooks rather than by unit tests.

## Citation

If you use `nmp_modeling`, please cite the software release (see `CITATION.cff`; a Zenodo DOI is minted for each tagged release) and the papers whose models it implements:

- Deco G, et al. Whole-brain multimodal neuroimaging model using serotonin receptor maps explains non-linear functional effects of LSD. *Current Biology* 2018. doi:10.1016/j.cub.2018.07.083
- Deco G, Sanz Perl Y, Vohryzek J, Luppi AI, Kringelbach ML. Neurotransmission-modulated whole-brain computation captures full task repertoire. *Cell Reports* 2026.
- Luppi AI, et al. Competitive interactions shape mammalian brain network dynamics and computation. *Nature Neuroscience* 2026.
- Markello RD, et al. neuromaps: structural and functional interpretation of brain maps. *Nature Methods* 2022. doi:10.1038/s41592-022-01625-w

## Acknowledgements

Development is supported by the NIH BRAIN Initiative (R01MH139565). The Neuronumba backend is developed by Gustavo Deco's group at Universitat Pompeu Fabra.

## License

Apache License 2.0. See `LICENSE`.
