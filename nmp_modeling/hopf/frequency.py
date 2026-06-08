import numpy as np
from scipy.ndimage import gaussian_filter1d

from nmp_modeling.preprocessing import make_bandpass_filter


def estimate_peak_frequency(
    timeseries,
    tr_seconds,
    band=(0.008, 0.08),
    order=2,
    smooth_sigma_hz=0.01,
    trim=0,
    normalize_by_variance=True,
):
    """Estimate regional Hopf frequencies from BOLD time series.

    The implementation follows the common Hopf/GEC FFT-peak practice:
      1. Demean, detrend, and band-pass filter each regional signal.
      2. Optionally trim edge samples after filtering.
      3. Compute the FFT power spectrum.
      4. Optionally divide each regional spectrum by filtered signal variance.
      5. Average power spectra across subjects.
      6. Smooth each regional spectrum with a Gaussian kernel.
      7. Select the peak-power frequency for each region.

    Input shape must be (time, nodes) or (subjects, time, nodes).
    Returned frequencies are in Hz.
    """
    arr = np.asarray(timeseries, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(
            "timeseries must have shape (time, nodes) or "
            "(subjects, time, nodes)."
        )
    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be positive.")
    trim = int(trim)
    if trim < 0:
        raise ValueError("trim must be non-negative.")

    n_subjects, n_time, n_nodes = arr.shape
    n_used = n_time - 2 * trim
    if n_used < 4:
        raise ValueError("Not enough time points remain after trimming.")

    bandpass = make_bandpass_filter(
        low=band[0],
        high=band[1],
        tr_seconds=tr_seconds,
        order=order,
        axis=0,
        apply_detrend=True,
        apply_demean=True,
        remove_artifacts=False,
    )

    freqs = np.arange(n_used // 2) / (n_used * float(tr_seconds))
    search = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(search):
        raise ValueError("No FFT frequency bins fall within the requested band.")

    power = np.zeros((freqs.size, n_nodes, n_subjects), dtype=float)
    for s in range(n_subjects):
        filtered = bandpass(arr[s])
        if trim > 0:
            filtered = filtered[trim:-trim]

        spectrum = np.fft.fft(filtered, axis=0)
        subject_power = np.abs(spectrum[: freqs.size]) ** 2 / (
            n_used / float(tr_seconds)
        )
        if normalize_by_variance:
            variance = np.var(filtered, axis=0, ddof=1)
            if np.any(variance <= 0):
                raise ValueError(
                    "Cannot normalize power spectrum with zero variance."
                )
            subject_power = subject_power / variance[None, :]
        power[:, :, s] = subject_power

    mean_power = np.mean(power, axis=2)

    if smooth_sigma_hz > 0:
        freq_step = freqs[1] - freqs[0]
        smooth_sigma_bins = smooth_sigma_hz / freq_step
        mean_power = gaussian_filter1d(
            mean_power,
            sigma=smooth_sigma_bins,
            axis=0,
            mode="constant",
            cval=0.0,
            truncate=np.sqrt(-2.0 * np.log(1e-6)),
        )

    band_power = mean_power[search, :]
    band_freqs = freqs[search]
    peak_indices = np.argmax(band_power, axis=0)

    return band_freqs[peak_indices]
