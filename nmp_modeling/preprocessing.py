import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt


def make_bandpass_filter(
    low=0.01,
    high=0.1,
    tr_seconds=2.0,
    order=2,
    axis=0,
    apply_detrend=True,
    apply_demean=True,
    remove_artifacts=True,
    artifact_threshold=3.0,
):
    """Create a Butterworth band-pass filter for time series data."""
    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be positive.")

    if order < 1:
        raise ValueError("order must be at least 1.")

    if artifact_threshold <= 0:
        raise ValueError("artifact_threshold must be positive.")

    sampling_rate = 1.0 / float(tr_seconds)
    nyquist = 0.5 * sampling_rate

    if not (0 < low < high < nyquist):
        raise ValueError("Frequencies must satisfy 0 < low < high < Nyquist.")

    sos = butter(
        N=order,
        Wn=[low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )

    def bandpass(data):
        """Apply the configured band-pass filter."""
        arr = np.asarray(data, dtype=float)

        if arr.ndim != 2:
            raise ValueError("data must be a 2D array.")

        if axis not in (0, 1, -1, -2):
            raise ValueError("axis must refer to one of the two data dimensions.")

        if np.isnan(arr).any():
            raise FloatingPointError("NaN found when applying band-pass filter.")

        out = np.array(arr, dtype=float, copy=True)

        if apply_detrend:
            out = detrend(out, axis=axis)

        if apply_demean:
            out = out - np.mean(out, axis=axis, keepdims=True)

        if remove_artifacts:
            sd = np.std(out, axis=axis, keepdims=True)
            limit = artifact_threshold * sd
            out = np.clip(out, -limit, limit)

        return sosfiltfilt(sos, out, axis=axis)

    return bandpass
