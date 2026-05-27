from dataclasses import dataclass
import numpy as np


VALID_INPUT_TYPES = {
    "timeseries",
    "fc",
    "fisher_z_fc",
    "observable",
}


@dataclass(frozen=True)
class DataInfo:
    """Shape metadata for empirical input data."""
    input_type: str
    n_subjects: int
    n_nodes: int | None = None
    n_timepoints: int | None = None
    has_subject_axis: bool = False


def infer_input_type(data):
    """Infer a likely input_type from array shape."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        if arr.shape[0] == arr.shape[1]:
            return "fc"
        return "timeseries"
    if arr.ndim == 3:
        if arr.shape[1] == arr.shape[2]:
            return "fc"
        return "timeseries"
    return "observable"


def infer_data_info(data, input_type):
    """Infer subject, node, and time dimensions from empirical input data."""
    if input_type not in VALID_INPUT_TYPES:
        raise ValueError(f"Unknown input_type: {input_type}")

    arr = np.asarray(data, dtype=float)

    if input_type == "timeseries":
        if arr.ndim == 2:
            n_timepoints, n_nodes = arr.shape
            _check_time_node_shape(n_timepoints, n_nodes)
            return DataInfo(
                input_type=input_type,
                n_subjects=1,
                n_nodes=n_nodes,
                n_timepoints=n_timepoints,
                has_subject_axis=False,
            )
        if arr.ndim == 3:
            n_subjects, n_timepoints, n_nodes = arr.shape
            _check_subject_count(n_subjects)
            _check_time_node_shape(n_timepoints, n_nodes)
            return DataInfo(
                input_type=input_type,
                n_subjects=n_subjects,
                n_nodes=n_nodes,
                n_timepoints=n_timepoints,
                has_subject_axis=True,
            )
        raise ValueError(
            "timeseries data must have shape (time, nodes) or "
            "(subjects, time, nodes)."
        )

    if input_type in {"fc", "fisher_z_fc"}:
        if arr.ndim == 2:
            _check_square_matrix_shape(arr.shape, input_type)
            return DataInfo(
                input_type=input_type,
                n_subjects=1,
                n_nodes=arr.shape[0],
                has_subject_axis=False,
            )
        if arr.ndim == 3:
            n_subjects = arr.shape[0]
            _check_subject_count(n_subjects)
            _check_square_matrix_shape(arr.shape[1:], input_type)
            return DataInfo(
                input_type=input_type,
                n_subjects=n_subjects,
                n_nodes=arr.shape[1],
                has_subject_axis=True,
            )
        raise ValueError(
            f"{input_type} data must have shape (nodes, nodes) or "
            "(subjects, nodes, nodes)."
        )

    return DataInfo(
        input_type=input_type,
        n_subjects=1,
        has_subject_axis=False,
    )


def subject_data(data, input_type, subject_index):
    """Return one subject's data if a subject axis is present."""
    info = infer_data_info(data, input_type)
    arr = np.asarray(data, dtype=float)

    if not info.has_subject_axis:
        return arr

    if subject_index is None:
        raise ValueError("subject_index is required for subject-wise empirical data.")
    if subject_index < 0 or subject_index >= info.n_subjects:
        raise IndexError(
            f"subject_index={subject_index} is out of range for "
            f"{info.n_subjects} subjects."
        )
    return arr[subject_index]


def _check_subject_count(n_subjects):
    """Check that at least one subject is present."""
    if n_subjects < 1:
        raise ValueError("Empirical data must contain at least one subject.")


def _check_time_node_shape(n_timepoints, n_nodes):
    """Check time-series dimensions."""
    if n_timepoints < 2:
        raise ValueError("Time series data must contain at least two time points.")
    if n_nodes < 2:
        raise ValueError("Time series data must contain at least two nodes.")


def _check_square_matrix_shape(shape, name):
    """Check matrix dimensions."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"{name} data must be square.")
    if shape[0] < 2:
        raise ValueError(f"{name} data must contain at least two nodes.")
