from dataclasses import dataclass, field

import numpy as np

from nmp_modeling import objectives
from nmp_modeling import observables
from nmp_modeling.data import infer_data_info, subject_data


VALID_INPUT_TYPES = {
    "timeseries",
    "fc",
    "fisher_z_fc",
    "observable",
}


def _merge_params(defaults, overrides):
    """Merge default parameters with user-specified overrides."""
    params = dict(defaults)
    params.update(overrides)
    return params


def _mean_arrays(values, **kwargs):
    """Average same-shaped arrays."""
    arrays = [np.asarray(v, dtype=float) for v in values]
    return np.mean(np.stack(arrays, axis=0), axis=0)


def _concat_arrays(values, **kwargs):
    """Concatenate flattened arrays."""
    arrays = [np.ravel(np.asarray(v, dtype=float)) for v in values]
    return np.concatenate(arrays)


def _aggregate_fc(values, fisher_z=False, average_fisher_z=True, **kwargs):
    """Aggregate subject-level FC matrices."""
    arrays = [np.asarray(v, dtype=float) for v in values]

    if not average_fisher_z:
        return np.mean(np.stack(arrays, axis=0), axis=0)

    if fisher_z:
        return np.mean(np.stack(arrays, axis=0), axis=0)

    z_arrays = [observables.fisher_z_matrix(v) for v in arrays]
    z_mean = np.mean(np.stack(z_arrays, axis=0), axis=0)

    return observables.inverse_fisher_z_matrix(z_mean)


def _aggregate_fc_edges(values, triangle="upper", **kwargs):
    """Aggregate FC edge-value distributions across subjects."""
    edges = []

    for value in values:
        arr = np.asarray(value, dtype=float)

        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            arr = observables.matrix_edges(arr, triangle=triangle)

        edges.append(np.ravel(arr))

    return np.concatenate(edges)


def _compute_fc(data, fisher_z=False, **kwargs):
    """Compute simulated FC."""
    return observables.compute_fc(data, fisher_z=fisher_z)


def _compute_gbc(data, fisher_z=False, **kwargs):
    """Compute simulated GBC."""
    return observables.compute_gbc(data, fisher_z=fisher_z)


def _compute_fcd(data, window_size=30, step=2, fisher_z=False, triangle="upper", **kwargs):
    """Compute simulated sliding-window FCD distribution."""
    return observables.compute_swfcd_distribution(
        data,
        window_size=window_size,
        step=step,
        fisher_z=fisher_z,
        triangle=triangle,
    )


def _compute_phfcd(
    data,
    discard_offset=10,
    pattern_size=3,
    triangle="upper",
    **kwargs,
):
    """Compute simulated phase-FCD distribution."""
    return observables.compute_phfcd_distribution(
        data,
        discard_offset=discard_offset,
        pattern_size=pattern_size,
        triangle=triangle,
    )


def _empirical_fc(data, input_type, fisher_z=False, **kwargs):
    """Build empirical FC from supported input types."""
    if input_type == "timeseries":
        return observables.compute_fc(data, fisher_z=fisher_z)

    if input_type == "fc":
        fc = observables.check_square_matrix(data, name="fc")
        return observables.fisher_z_matrix(fc) if fisher_z else fc

    if input_type == "fisher_z_fc":
        zfc = observables.check_square_matrix(data, name="fisher_z_fc")
        return zfc if fisher_z else observables.inverse_fisher_z_matrix(zfc)

    if input_type == "observable":
        return np.asarray(data, dtype=float)

    raise ValueError(f"Unsupported input_type for FC observable: {input_type}")


def _empirical_gbc(data, input_type, fisher_z=False, **kwargs):
    """Build empirical GBC from supported input types."""
    if input_type == "timeseries":
        return observables.compute_gbc(data, fisher_z=fisher_z)

    if input_type == "fc":
        fc = observables.check_square_matrix(data, name="fc")
        fc = observables.fisher_z_matrix(fc) if fisher_z else fc
        return observables.compute_gbc_from_fc(fc)

    if input_type == "fisher_z_fc":
        zfc = observables.check_square_matrix(data, name="fisher_z_fc")
        fc = zfc if fisher_z else observables.inverse_fisher_z_matrix(zfc)
        return observables.compute_gbc_from_fc(fc)

    if input_type == "observable":
        return np.asarray(data, dtype=float)

    raise ValueError(f"Unsupported input_type for GBC observable: {input_type}")


def _empirical_fcd(data, input_type, **params):
    """Build empirical sliding-window FCD distribution."""
    if input_type == "timeseries":
        return _compute_fcd(data, **params)

    if input_type == "observable":
        return np.asarray(data, dtype=float)

    raise ValueError(
        "FCD observables require input_type='timeseries' or input_type='observable'."
    )


def _empirical_phfcd(data, input_type, **params):
    """Build empirical phase-FCD distribution."""
    if input_type == "timeseries":
        return _compute_phfcd(data, **params)

    if input_type == "observable":
        return np.asarray(data, dtype=float)

    raise ValueError(
        "phFCD observables require input_type='timeseries' or input_type='observable'."
    )


def _distance_fc_similarity(simulated, empirical, triangle="upper", metric="pearson", **kwargs):
    """Compare FC matrices by edge-wise similarity."""
    return objectives.edge_similarity_distance(
        simulated,
        empirical,
        triangle=triangle,
        metric=metric,
    )


def _distance_vector_similarity(simulated, empirical, metric="pearson", **kwargs):
    """Compare vectors by Pearson or cosine similarity."""
    return objectives.similarity_distance(
        simulated,
        empirical,
        metric=metric,
    )


def _distance_frobenius(simulated, empirical, **kwargs):
    """Compare arrays by Frobenius distance."""
    return objectives.frobenius_distance(simulated, empirical)


def _distance_mse(simulated, empirical, **kwargs):
    """Compare arrays by mean squared error."""
    return objectives.mse_distance(simulated, empirical)


def _distance_mean_abs_diff(simulated, empirical, triangle="upper", **kwargs):
    """Compare mean values."""
    return objectives.mean_abs_difference(
        simulated,
        empirical,
        triangle=triangle,
    )


def _distance_fc_distribution_ks(simulated, empirical, triangle="upper", **kwargs):
    """Compare FC edge-value distributions by KS distance."""
    return objectives.fc_distribution_ks_distance(
        simulated,
        empirical,
        triangle=triangle,
    )


def _distance_ks(simulated, empirical, **kwargs):
    """Compare distributions by KS distance."""
    return objectives.ks_distance(simulated, empirical)


@dataclass
class ObservableSpec:
    """Definition of one observable label."""
    name: str
    compute_fn: object
    empirical_fn: object
    distance_fn: object
    aggregate_fn: object
    defaults: dict = field(default_factory=dict)
    allowed_input_types: tuple = ("timeseries", "observable")

    def resolve_params(self, overrides):
        """Return default parameters updated by user overrides."""
        return _merge_params(self.defaults, overrides)

    def validate_input_type(self, input_type):
        """Check whether this observable supports the empirical input type."""
        if input_type not in VALID_INPUT_TYPES:
            raise ValueError(f"Unknown input_type: {input_type}")

        if input_type not in self.allowed_input_types:
            raise ValueError(
                f"Observable '{self.name}' does not support input_type='{input_type}'."
            )


@dataclass
class EmpiricalTarget:
    """Empirical observable with cached value and matched simulation rules."""
    spec: ObservableSpec
    data: object
    input_type: str
    params: dict = field(default_factory=dict)
    label: str = None
    preprocess: object = None
    _data_info: object = field(default=None, init=False, repr=False)
    _value: object = field(default=None, init=False, repr=False)
    _values_by_count: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize target label and empirical data metadata."""
        if self.label is None:
            self.label = self.spec.name
        self._data_info = infer_data_info(self.data, self.input_type)

    @property
    def n_subjects(self):
        """Return the number of empirical subjects."""
        return self._data_info.n_subjects

    @property
    def n_nodes(self):
        """Return the number of nodes if known."""
        return self._data_info.n_nodes

    @property
    def n_timepoints(self):
        """Return the number of time points if known."""
        return self._data_info.n_timepoints

    @property
    def is_subjectwise(self):
        """Return whether empirical data has an explicit subject axis."""
        return self._data_info.has_subject_axis

    def _preprocess(self, data):
        """Apply target-level preprocessing to time series data."""
        if self.preprocess is None:
            return data
        arr = np.asarray(data, dtype=float)
        return self.preprocess(arr.copy())

    def _empirical_observable_for_subject(self, subject_index):
        """Compute one subject-level empirical observable."""
        data = subject_data(self.data, self.input_type, subject_index)
        if self.input_type == "timeseries":
            data = self._preprocess(data)
        return self.spec.empirical_fn(
            data,
            self.input_type,
            **self.params,
        )

    def empirical_value(self, n_subjects=None):
        """Return the aggregated empirical observable value."""
        self.spec.validate_input_type(self.input_type)

        if not self.is_subjectwise:
            if self._value is None:
                data = (
                    self._preprocess(self.data)
                    if self.input_type == "timeseries"
                    else self.data
                )
                self._value = self.spec.empirical_fn(
                    data,
                    self.input_type,
                    **self.params,
                )
            return self._value

        count = self.n_subjects if n_subjects is None else int(n_subjects)
        if count < 1:
            raise ValueError("n_subjects must be at least 1.")
        if count > self.n_subjects:
            raise ValueError(
                f"Requested {count} subjects, but empirical target "
                f"'{self.label}' only has {self.n_subjects}."
            )
        if count not in self._values_by_count:
            values = [
                self._empirical_observable_for_subject(i)
                for i in range(count)
            ]
            self._values_by_count[count] = self.spec.aggregate_fn(
                values,
                **self.params,
            )
        return self._values_by_count[count]

    @property
    def value(self):
        """Return cached empirical observable value using all empirical subjects."""
        return self.empirical_value()

    @property
    def empirical_target(self):
        """Alias for fitting code."""
        return self.value

    def observable(self, simulated_data):
        """Compute the matching observable from simulated data."""
        data = self._preprocess(simulated_data)
        return self.spec.compute_fn(
            data,
            **self.params,
        )

    def aggregate_observable(self, simulated_values):
        """Aggregate subject-level simulated observables."""
        return self.spec.aggregate_fn(
            simulated_values,
            **self.params,
        )

    def distance(self, simulated_value, empirical_value=None):
        """Compute distance between simulated and empirical observables."""
        if empirical_value is None:
            empirical_value = self.value

        return self.spec.distance_fn(
            simulated_value,
            empirical_value,
            **self.params,
        )


def _fc_defaults(fisher_z=False):
    """Return default parameters for FC-family observables."""
    return {
        "fisher_z": fisher_z,
        "triangle": "upper",
        "metric": "pearson",
        "average_fisher_z": True,
    }


def _fcd_defaults():
    """Return default parameters for sliding-window FCD."""
    return {
        "window_size": 30,
        "step": 2,
        "fisher_z": False,
        "triangle": "upper",
    }


OBSERVABLE_SPECS = {
    "fc_corr": ObservableSpec(
        name="fc_corr",
        compute_fn=_compute_fc,
        empirical_fn=_empirical_fc,
        distance_fn=_distance_fc_similarity,
        aggregate_fn=_aggregate_fc,
        defaults=_fc_defaults(fisher_z=False),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "fc_corr_z": ObservableSpec(
        name="fc_corr_z",
        compute_fn=_compute_fc,
        empirical_fn=_empirical_fc,
        distance_fn=_distance_fc_similarity,
        aggregate_fn=_aggregate_fc,
        defaults=_fc_defaults(fisher_z=True),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "gbc_corr": ObservableSpec(
        name="gbc_corr",
        compute_fn=_compute_gbc,
        empirical_fn=_empirical_gbc,
        distance_fn=_distance_vector_similarity,
        aggregate_fn=_mean_arrays,
        defaults=_fc_defaults(fisher_z=False),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "gbc_corr_z": ObservableSpec(
        name="gbc_corr_z",
        compute_fn=_compute_gbc,
        empirical_fn=_empirical_gbc,
        distance_fn=_distance_vector_similarity,
        aggregate_fn=_mean_arrays,
        defaults=_fc_defaults(fisher_z=True),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "fc_frobenius": ObservableSpec(
        name="fc_frobenius",
        compute_fn=_compute_fc,
        empirical_fn=_empirical_fc,
        distance_fn=_distance_frobenius,
        aggregate_fn=_aggregate_fc,
        defaults=_fc_defaults(fisher_z=False),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "fc_mse": ObservableSpec(
        name="fc_mse",
        compute_fn=_compute_fc,
        empirical_fn=_empirical_fc,
        distance_fn=_distance_mse,
        aggregate_fn=_aggregate_fc,
        defaults=_fc_defaults(fisher_z=False),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "fc_mean_abs_diff": ObservableSpec(
        name="fc_mean_abs_diff",
        compute_fn=_compute_fc,
        empirical_fn=_empirical_fc,
        distance_fn=_distance_mean_abs_diff,
        aggregate_fn=_aggregate_fc,
        defaults=_fc_defaults(fisher_z=False),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "fc_distribution_ks": ObservableSpec(
        name="fc_distribution_ks",
        compute_fn=_compute_fc,
        empirical_fn=_empirical_fc,
        distance_fn=_distance_fc_distribution_ks,
        aggregate_fn=_aggregate_fc_edges,
        defaults=_fc_defaults(fisher_z=False),
        allowed_input_types=("timeseries", "fc", "fisher_z_fc", "observable"),
    ),
    "fcd_ks": ObservableSpec(
        name="fcd_ks",
        compute_fn=_compute_fcd,
        empirical_fn=_empirical_fcd,
        distance_fn=_distance_ks,
        aggregate_fn=_concat_arrays,
        defaults=_fcd_defaults(),
        allowed_input_types=("timeseries", "observable"),
    ),
    "phfcd_ks": ObservableSpec(
        name="phfcd_ks",
        compute_fn=_compute_phfcd,
        empirical_fn=_empirical_phfcd,
        distance_fn=_distance_ks,
        aggregate_fn=_concat_arrays,
        defaults={
            "discard_offset": 10,
            "pattern_size": 3,
            "triangle": "upper",
        },
        allowed_input_types=("timeseries", "observable"),
    ),
}


def get_observable_spec(name):
    """Return an observable spec by name."""
    if name not in OBSERVABLE_SPECS:
        available = ", ".join(sorted(OBSERVABLE_SPECS))
        raise ValueError(f"Unknown observable '{name}'. Available: {available}")

    return OBSERVABLE_SPECS[name]


def make_empirical_target(
    data,
    observable,
    input_type="timeseries",
    label=None,
    preprocess=None,
    **params,
):
    """Create one empirical target from data and an observable label."""
    if isinstance(observable, dict):
        observable_params = dict(observable)
        spec_name = observable_params.pop("name")
        dict_label = observable_params.pop("label", None)
        dict_preprocess = observable_params.pop("preprocess", None)

        if label is None:
            label = dict_label
        elif dict_label is not None and dict_label != label:
            raise ValueError("Conflicting target labels from label and observable['label'].")

        if preprocess is None:
            preprocess = dict_preprocess
        elif dict_preprocess is not None and dict_preprocess is not preprocess:
            raise ValueError(
                "Conflicting preprocessing functions from preprocess and observable['preprocess']."
            )

        observable_params.update(params)
    else:
        spec_name = observable
        observable_params = params

    spec = get_observable_spec(spec_name)
    resolved_params = spec.resolve_params(observable_params)
    spec.validate_input_type(input_type)

    return EmpiricalTarget(
        spec=spec,
        data=data,
        input_type=input_type,
        params=resolved_params,
        label=label,
        preprocess=preprocess,
    )


def make_empirical_targets(data, observables, input_type="timeseries", **shared_params):
    """Create multiple empirical targets from the same empirical data."""
    targets = []

    for observable in observables:
        if isinstance(observable, dict):
            params = dict(shared_params)
            params.update(observable)
            targets.append(
                make_empirical_target(
                    data=data,
                    observable=params,
                    input_type=input_type,
                )
            )
        else:
            targets.append(
                make_empirical_target(
                    data=data,
                    observable=observable,
                    input_type=input_type,
                    **shared_params,
                )
            )

    return targets
