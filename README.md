# nmp_modeling

**A connector between Neuromaps-PRIME brain maps and whole-brain simulation packages.**

---

## Status

This is a v1 vertical slice — a minimal end-to-end implementation intended to validate the API design before further work. **Feedback is welcome and expected.** See [What's intentionally missing from v1](#whats-intentionally-missing-from-v1) below.

## What this is

`nmp_modeling` is a thin Python connector that lets you take a brain map (e.g., a GABA receptor density map from Neuromaps-PRIME) and feed it as a *spatial prior* into a whole-brain simulation (e.g., a Deco-style neural mass model in [Neuronumba](https://github.com/neich/neuronumba)). The user describes how the brain map enters the model via a string expression like `"a + b * gaba"`, where `a` and `b` are free scalar parameters to be fit. The library handles the parsing, the parameter sweep, and the per-backend translation.

The job of this library is to connect Neuromaps-PRIME to existing simulation tools — each modeling framework gets its own backend adapter and the simulation itself is delegated.


## Quickstart

### Install

The package isn't on PyPI yet. From a clone of this repo:

```bash
pip install -e .
```

To use the Neuronumba adapter, you also need Neuronumba installed:

```bash
pip install -e "git+https://github.com/neich/neuronumba.git#egg=neuronumba&subdirectory=src"
```

### Illustrative example

```python
import numpy as np
from nmp_modeling import (
    MapParametrization, FreeParam,
    observables, grid_sweep,
)
from nmp_modeling.adapters.neuronumba import NeuronumbaAdapter

# 1. Describe how a brain map enters the model.
p_we = MapParametrization(
    target="w_gain_e",                  # name of the model attribute
    expression="a + b * receptor",      # how the map enters
    maps={"receptor": gaba_map},        # numpy array, one value per node
    free_params={
        "a": FreeParam(init=0.0, bounds=(-0.2, 0.5)),
        "b": FreeParam(init=0.0, bounds=(-0.2, 0.5)),
    },
)

# 2. Build a backend-specific adapter.
adapter = NeuronumbaAdapter(
    weights=SC,
    parametrizations=[p_we],
    fixed_model_attrs={"auto_fic": True},
)

# 3. Compute empirical observables once.
emp_fcd = observables.swfcd(empirical_bold)

# 4. Sweep.
result = grid_sweep(
    adapter=adapter,
    free_grid={"G": np.arange(0, 2.5, 0.25), "a": [0.0], "b": [0.0]},
    fixed={},
    observable=observables.swfcd,
    empirical_target=emp_fcd,
    distance=observables.ks_distance,
    n_subjects=15,
    run_seeds=[11, 23, 37],
)
print(result.best_theta, result.best_loss)
```

For an end-to-end example that mirrors a real research workflow, see [`examples/example_neuronumba.ipynb`](examples/example_neuronumba.ipynb).

## Architecture

The library is organised into three layers, each with a narrow job. Reviewers should evaluate whether the boundaries between layers are sensible and whether each layer is doing the right amount of work.

### Layer 1: `MapParametrization` (simulator-agnostic)

A user-facing dataclass that ties together (a) a target variable in some model's equations, (b) a string expression like `"a + b * gaba"`, (c) the brain maps referenced in that expression, and (d) a set of *free* and *fixed* scalar parameters. It knows how to evaluate itself at given parameter values to produce a per-node numpy vector.

The expression is parsed by [SymPy](https://www.sympy.org/) (safely — `sympy.sympify` does not exec arbitrary Python) and JIT-compiled to a numpy callable on first use. SymPy gives us free parsing, free symbolic differentiation if we ever want gradient-based fitting, and elementwise broadcasting for free.

Lives in [`src/nmp_modeling/parametrization.py`](src/nmp_modeling/parametrization.py).

### Layer 2: Backend adapters (simulator-specific)

A thin translation layer per simulation backend. Each adapter takes a structural connectivity matrix and a list of `MapParametrization`s and exposes a single method:

```python
adapter.simulate(theta: dict, seed: int) -> bold_array
```

That's the entire protocol. Adapters are responsible for translating between the `(theta, parametrizations)` worldview and the specific API of one simulation package. The Neuronumba adapter, for instance, calls `model.set_attributes(...)` and `model.configure(weights=..., g=...)` and then delegates to `simulate_nodelay(...)`.

The Neuronumba adapter is the only file in v1 that imports Neuronumba.

Lives in [`src/nmp_modeling/adapters/neuronumba.py`](src/nmp_modeling/adapters/neuronumba.py).

### Layer 3: Fitting (backend-agnostic)

Knows nothing about Neuronumba. Treats `adapter.simulate` as a black box and runs a parameter sweep. V1 ships with grid sweep only; CMA-ES and other optimisers are planned (see [Roadmap](#roadmap)).

The protocol — multi-seed averaging, multi-subject simulation, per-subject observable aggregation — is opinionated and matches the pattern used in Deco-style modelling:

```
For each grid point theta:
    For each run_seed:
        For each of n_subjects synthetic subjects:
            bold = adapter.simulate(theta, subject_seed)
            obs[i] = observable(bold)
        loss[seed] = distance(aggregate(obs), empirical_target)
    mean_loss[theta] = mean(loss across seeds)
```

Lives in [`src/nmp_modeling/fitting.py`](src/nmp_modeling/fitting.py) and [`src/nmp_modeling/observables.py`](src/nmp_modeling/observables.py).

## How the layers interact

A subtle point worth tracing explicitly. When you call `grid_sweep(...)`, it iterates over the Cartesian product of `free_grid`, builds a flat `theta` dict for each grid point (e.g., `{"a": 0.1, "b": 0.05, "G": 1.5}`), and passes that dict to `adapter.simulate(theta, seed)`. The grid sweep itself never sees the parametrization expression or the brain maps.

The expression evaluation happens *inside* the adapter:

```python
# Inside NeuronumbaAdapter.simulate():
for p in self.parametrizations:
    free_vals = {k: theta[k] for k in p.free_params if k in theta}
    attrs[p.target] = p.evaluate(free_vals)   # <-- the expression runs here
model.set_attributes(attrs)
```

This factoring is deliberate: the fitting layer needs to be reusable across backends, and a future Kuramoto adapter (writing into `omega`) or REACT integration (no `theta` at all) will use parametrizations completely differently. By making the adapter the single point of translation, the fitting layer stays clean.

The implication: parameter names in `free_grid` must match parameter names in `MapParametrization.free_params` exactly. Mismatches raise `KeyError` at simulation time.

## Repository layout

```
nmp-modeling/
├── README.md                      <-- you are here
├── LICENSE
├── pyproject.toml                 <-- packaging config
├── .gitignore
├── nmp_modeling/
│  ├── __init__.py            <-- public API
│  ├── parametrization.py     <-- Layer 1
│  ├── observables.py         <-- Layer 3a
│  ├── fitting.py             <-- Layer 3b (grid_sweep)
│  └── adapters/
│           ├── __init__.py
│           └── neuronumba.py      <-- Layer 2
├── tests/
│   ├── test_parametrization.py
│   ├── test_observables.py
│   └── test_fitting.py
├── examples/
│   └── example_neuronumba.ipynb   <-- end-to-end usage
└── docs/
    ├── DESIGN.md                  <-- design rationale
    └── ROADMAP.md                 <-- what's next
```

## Running the tests

Tests cover the parametrization layer, the observables, and the fitting layer (using a mock adapter so Neuronumba is not required). They do not cover the Neuronumba adapter itself — that requires a real Neuronumba installation and is best validated by running [`examples/example_neuronumba.ipynb`](examples/example_neuronumba.ipynb) end-to-end.

```bash
pip install pytest
pytest tests/
```

All 21 tests should pass.

## Roadmap

In rough priority order — feedback on this ordering is welcome.

1. **Validate the v1 abstraction** by running `examples/example_neuronumba.ipynb` against a real Neuronumba install, with the actual `serotonin2A.Deco2018` model. The notebook is structured so any breakage will be in the Neuronumba adapter, not in the parametrization or fitting layers.
2. **Build a second adapter** (Kuramoto is the natural choice). The right test of the abstraction is that we can build it without modifying `MapParametrization` or `grid_sweep`.
3. **Add CMA-ES and/or differential evolution** alongside `grid_sweep`. Same `(adapter, observable, empirical_target, distance)` signature; just a different fitting primitive. Required before tackling Zhang-2024-style 10-parameter fits.
4. **Hook into Neuromaps-PRIME's fetch API** so users go from `nmp.fetch("gaba_schaefer500")` straight into a `MapParametrization`.
5. **Add a REACT integration** as a separate top-level function (likely `nmp_modeling.dual_regression(maps, bold) -> weights`). Worth designing carefully because it's the place the adapter abstraction does *not* fit, and we want to make sure that's honest in the API rather than papered over.

## Design questions for review

If you're reading this with feedback in mind, here are the choices I'd most like to hear about:

- **Is the three-layer split right?** Specifically: should the adapter own more of the fitting protocol (e.g., the multi-seed averaging) or less? Currently the adapter is a single-shot `simulate(theta, seed)` and the fitting layer owns everything else.

- **Multi-stage threading via `fixed`.** Currently you hand the previous stage's `result.params` into the next stage's `fixed` argument. This is bare-bones — any objections, or any features you'd want here (e.g., a `MultiStageFit` helper that records the staging history)?

## Citation

To be added on first publication.

## License

Apache-2.0 (provisional). See LICENSE.
