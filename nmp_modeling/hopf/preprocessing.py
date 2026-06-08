import numpy as np

from nmp_modeling.preprocessing import make_bandpass_filter


def ensure_subject_time_node(timeseries):
    """Return time series as (subjects, time, nodes)."""
    arr = np.asarray(timeseries, dtype=float)
    if arr.ndim == 2:
        return arr[None, :, :], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(
        "timeseries must have shape (time, nodes) or "
        "(subjects, time, nodes)."
    )


def restore_time_series_rank(timeseries, was_single_subject):
    """Restore a preprocessed array to its original single/multi-subject rank."""
    arr = np.asarray(timeseries, dtype=float)
    if was_single_subject:
        return arr[0]
    return arr


def trim_time_series(timeseries, trim=0):
    """Trim edge samples from a time-by-node time series."""
    trim = int(trim)
    if trim < 0:
        raise ValueError("trim must be non-negative.")
    if trim == 0:
        return np.asarray(timeseries, dtype=float)
    arr = np.asarray(timeseries, dtype=float)
    if arr.shape[0] <= 2 * trim:
        raise ValueError("Not enough time points remain after trimming.")
    return arr[trim:-trim]


def preprocess_time_series(
    timeseries,
    tr_seconds,
    band=(0.008, 0.08),
    order=2,
    trim=0,
    remove_artifacts=False,
    artifact_threshold=3.0,
):
    """Prepare Hopf/BOLD-like time series for FC, lagged observables, and GEC."""
    arr, was_single_subject = ensure_subject_time_node(timeseries)

    bandpass = make_bandpass_filter(
        low=band[0],
        high=band[1],
        tr_seconds=tr_seconds,
        order=order,
        axis=0,
        apply_detrend=True,
        apply_demean=True,
        remove_artifacts=remove_artifacts,
        artifact_threshold=artifact_threshold,
    )

    out = []
    for s in range(arr.shape[0]):
        filtered = bandpass(arr[s])
        filtered = trim_time_series(filtered, trim=trim)
        out.append(filtered)

    out = np.asarray(out, dtype=float)
    return restore_time_series_rank(out, was_single_subject)
