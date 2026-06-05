import numpy as np
from scipy.ndimage import gaussian_filter1d

from nmp_modeling.preprocessing import make_bandpass_filter


def estimate_peak_frequency(
    timeseries,
    tr_seconds,
    band=(0.008, 0.08),
    order=2,
    smooth_sigma_hz=0.01,
):
    """Estimate regional Hopf frequencies from BOLD time series.

    This follows the common Hopf/GEC practice used in the Luppi et al.
    competitive-cooperative Hopf code:
      1. Demean, detrend, and band-pass filter each regional signal.
      2. Compute the FFT power spectrum.
      3. Average power spectra across subjects.
      4. Smooth each regional spectrum with a Gaussian kernel.
      5. Select the peak-power frequency for each region.

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

    n_subjects, n_time, n_nodes = arr.shape

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

    freqs = np.arange(n_time // 2) / (n_time * float(tr_seconds))
    search = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(search):
        raise ValueError("No FFT frequency bins fall within the requested band.")

    power = np.zeros((freqs.size, n_nodes, n_subjects), dtype=float)
    for s in range(n_subjects):
        filtered = bandpass(arr[s])
        spectrum = np.fft.fft(filtered, axis=0)
        power[:, :, s] = np.abs(spectrum[: freqs.size]) ** 2 / (
            n_time / float(tr_seconds)
        )
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
