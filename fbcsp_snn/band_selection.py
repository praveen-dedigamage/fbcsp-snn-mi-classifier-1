"""Adaptive frequency band selection via Fisher discriminant ratio.

Algorithm
---------
1. Estimate PSD per trial with Welch's method, average over channels.
2. Compute multi-class Fisher ratio at every frequency bin.
3. Score dense candidate bands (fixed width, fixed step) by integrating the
   Fisher ratio over each band's passband.
4. Greedily select the top-K candidates while enforcing ≤ 50 % overlap between
   any two selected bands.

The routine is designed to run **only on training data** inside each CV fold so
that no information from the validation / test split leaks into band selection.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, welch

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_bands(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_bands: int = 6,
    bandwidth: float = 4.0,
    step: float = 2.0,
    band_range: Tuple[float, float] = (4.0, 40.0),
    min_fisher_fraction: float = 0.05,
    top_k_channels: Optional[int] = None,
) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
    """Select the best *n_bands* frequency bands from training data.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        Class labels, shape ``(n_trials,)``.  May be 1-indexed.
    sfreq : float
        Sampling frequency in Hz.
    n_bands : int
        Maximum number of bands to select.
    bandwidth : float
        Width of each candidate band in Hz.
    step : float
        Step size between candidate band centres in Hz.
    band_range : Tuple[float, float]
        Frequency range ``(f_min, f_max)`` for candidate generation.
    min_fisher_fraction : float
        Minimum Fisher score as a fraction of the top candidate score.
        Candidates below ``top_score * min_fisher_fraction`` are rejected,
        preventing noise bands from being forced in when ``n_bands`` is large.
        Default 0.05 (5 % of top score).

    Returns
    -------
    selected_bands : List[Tuple[float, float]]
        Ordered list of ``(lo, hi)`` frequency band boundaries in Hz.
    fisher_freqs : np.ndarray
        Frequency axis for the Fisher curve, shape ``(n_freqs,)``.
    fisher_curve : np.ndarray
        Fisher discriminant ratio at each frequency bin, shape ``(n_freqs,)``.
    top_k_channels : int, optional
        If set, rank channels by their peak Fisher ratio and average PSD
        over only the top-K before computing the Fisher curve.  ``None``
        (default) uses all channels — current behaviour.
    """
    fisher_freqs, fisher_curve = _compute_fisher_curve(
        X, y, sfreq, top_k_channels=top_k_channels
    )

    candidates = _generate_candidates(bandwidth, step, band_range)
    logger.info(
        "Band selection: %d candidate bands  (bw=%.1f Hz, step=%.1f Hz, range=%s)",
        len(candidates), bandwidth, step, band_range,
    )

    scores = _score_candidates(candidates, fisher_freqs, fisher_curve)

    selected = _greedy_select(
        candidates, scores, n_bands, bandwidth,
        min_fisher_fraction=min_fisher_fraction,
    )

    logger.info("Selected %d bands: %s", len(selected), selected)
    return selected, fisher_freqs, fisher_curve


def select_bands_peak(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_bands: int = 6,
    bandwidth: float = 4.0,
    band_range: Tuple[float, float] = (4.0, 40.0),
    min_peak_distance_hz: float = 2.0,
    min_fisher_fraction: float = 0.05,
    top_k_channels: Optional[int] = None,
) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
    """Select bands by centering them on local Fisher-ratio peaks.

    Unlike :func:`select_bands`, which scores a fixed dense grid and enforces
    a maximum overlap constraint, this function:

    1. Finds local maxima (peaks) in the Fisher discriminant ratio curve.
    2. Centers a band of *bandwidth* Hz on each peak.
    3. Allows full overlap between bands — two peaks close together will
       produce partially overlapping bands, which is intentional because
       both capture genuinely discriminative signal.

    A light 2 Hz moving-average smooth is applied before peak detection to
    suppress noisy micro-peaks without shifting the true peak centres.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        Class labels (1-indexed), shape ``(n_trials,)``.
    sfreq : float
        Sampling frequency in Hz.
    n_bands : int
        Maximum number of bands to select.
    bandwidth : float
        Width of each band centred on a peak, in Hz.
    band_range : Tuple[float, float]
        ``(f_min, f_max)`` search range in Hz.
    min_peak_distance_hz : float
        Minimum separation between two selected peak centres in Hz.
        Prevents the algorithm from picking two peaks on the same ERD
        feature that happen to sit adjacent frequency bins.
    min_fisher_fraction : float
        Minimum Fisher score as a fraction of the top-peak score.
        Peaks below ``top_score * min_fisher_fraction`` are dropped,
        preventing noise bumps from being used as bands.

    Returns
    -------
    selected_bands : List[Tuple[float, float]]
        ``(lo, hi)`` band boundaries sorted by centre frequency.
    fisher_freqs : np.ndarray
        Frequency axis for the Fisher curve.
    fisher_curve : np.ndarray
        Fisher discriminant ratio per frequency bin.
    top_k_channels : int, optional
        If set, rank channels by peak Fisher and average PSD over only the
        top-K before computing the Fisher curve.  ``None`` uses all channels.
    """
    fisher_freqs, fisher_curve = _compute_fisher_curve(
        X, y, sfreq, top_k_channels=top_k_channels
    )

    fmin, fmax = band_range
    mask = (fisher_freqs >= fmin) & (fisher_freqs <= fmax)
    f    = fisher_freqs[mask]
    v    = fisher_curve[mask]

    if len(f) < 2:
        logger.warning("select_bands_peak: fewer than 2 frequency bins in range %s", band_range)
        return [(fmin, fmin + bandwidth)], fisher_freqs, fisher_curve

    freq_res = float(f[1] - f[0])              # Hz per bin

    # Light smoothing to merge adjacent micro-peaks (~2 Hz window)
    smooth_bins = max(1, round(2.0 / freq_res))
    v_smooth    = uniform_filter1d(v, size=smooth_bins)

    # Minimum inter-peak distance in bins
    dist_bins = max(1, round(min_peak_distance_hz / freq_res))

    peak_idxs, _ = find_peaks(v_smooth, distance=dist_bins)

    if len(peak_idxs) == 0:
        # Perfectly flat curve (e.g. non-responder subject) — use global max
        logger.warning(
            "select_bands_peak: no local peaks found in Fisher curve — "
            "using global maximum as single peak."
        )
        peak_idxs = np.array([int(np.argmax(v_smooth))])

    # Rank peaks by original (un-smoothed) Fisher value
    peak_idxs = peak_idxs[np.argsort(v[peak_idxs])[::-1]]

    # Drop peaks below relative Fisher threshold
    threshold = float(v[peak_idxs[0]]) * min_fisher_fraction
    peak_idxs = peak_idxs[v[peak_idxs] >= threshold]

    if len(peak_idxs) < n_bands:
        logger.warning(
            "select_bands_peak: only %d peaks above threshold (requested %d).",
            len(peak_idxs), n_bands,
        )

    # Take top-n_bands peaks
    peak_idxs = peak_idxs[:n_bands]

    # Build bands: centre ± bandwidth/2, clipped to band_range
    half = bandwidth / 2.0
    bands: List[Tuple[float, float]] = []
    for pidx in peak_idxs:
        centre = float(f[pidx])
        lo = max(fmin, centre - half)
        hi = min(fmax, centre + half)
        # Ensure minimum width (edge case: centre near fmin or fmax)
        if hi - lo < bandwidth * 0.5:
            if lo == fmin:
                hi = min(fmax, lo + bandwidth)
            else:
                lo = max(fmin, hi - bandwidth)
        bands.append((round(lo, 2), round(hi, 2)))

    # Sort by centre frequency (ascending) for reproducible display
    bands.sort(key=lambda b: (b[0] + b[1]) / 2.0)

    logger.info(
        "Peak-based band selection: %d bands centred on Fisher peaks "
        "(bandwidth=%.1f Hz, overlap allowed): %s",
        len(bands), bandwidth, bands,
    )
    return bands, fisher_freqs, fisher_curve


def select_channel_specific_bands(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_bands: int = 6,
    bandwidth: float = 4.0,
    band_range: Tuple[float, float] = (4.0, 40.0),
    nperseg: int = 256,
    smooth_hz: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find per-channel peak Fisher frequencies for channel-specific filtering.

    Rather than selecting a single set of frequency bands shared by all
    channels, this function identifies the top-``n_bands`` discriminative
    frequency peaks **independently for each EEG channel**.  The result is a
    ``(n_channels, n_bands)`` matrix of centre frequencies that feeds
    :func:`~fbcsp_snn.preprocessing.apply_channel_specific_filterbank` to
    construct a channel-personalised FBCSP decomposition.

    The number of resulting filter-bank slots is identical to the standard
    FBCSP (``n_bands``), so the downstream CSP fitting code is unchanged.
    The difference is that in band slot *b*, channel *c* is filtered at its
    own *b*-th best frequency rather than a global shared band.

    Parameters
    ----------
    X : np.ndarray
        Training EEG, shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        Class labels (1-indexed), shape ``(n_trials,)``.
    sfreq : float
        Sampling frequency in Hz.
    n_bands : int
        Number of band slots (peaks) to identify per channel.
    bandwidth : float
        Filter bandwidth in Hz; also used as the minimum peak separation
        distance to prevent two slots being assigned the same peak.
    band_range : Tuple[float, float]
        ``(f_min, f_max)`` frequency search range in Hz.
    nperseg : int
        Welch segment length (clipped to ``n_samples`` if larger).
    smooth_hz : float
        Smoothing window width in Hz applied to each channel's Fisher curve
        before peak detection.

    Returns
    -------
    channel_freqs : np.ndarray
        Centre frequencies, shape ``(n_channels, n_bands)``.
        Column 0 = strongest peak per channel, column 1 = second strongest, etc.
    fisher_per_ch : np.ndarray
        Raw (unsmoothed) per-channel Fisher ratio,
        shape ``(n_channels, n_freqs)``.
    freqs : np.ndarray
        Full frequency axis, shape ``(n_freqs,)``.
    """
    n_trials, n_channels, n_samples = X.shape
    nperseg = min(nperseg, n_samples)

    # --- Vectorised Welch PSD ---
    X_2d = X.reshape(n_trials * n_channels, n_samples)
    freqs, psd_2d = welch(X_2d, fs=sfreq, nperseg=nperseg, axis=-1)
    psd_3d = psd_2d.reshape(n_trials, n_channels, -1)  # (n_trials, n_ch, n_freqs)

    # --- Per-channel Fisher discriminant ratio ---
    fisher_per_ch = _fisher_ratio_per_channel(psd_3d, y)   # (n_ch, n_freqs)

    # --- Restrict to band_range ---
    lo_hz, hi_hz = band_range
    freq_mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    freqs_r = freqs[freq_mask]                              # (n_freqs_r,)
    fisher_r = fisher_per_ch[:, freq_mask]                  # (n_ch, n_freqs_r)

    if len(freqs_r) < 2:
        logger.warning(
            "select_channel_specific_bands: fewer than 2 frequency bins in "
            "range %s — returning uniform fallback bands.", band_range,
        )
        centre = (lo_hz + hi_hz) / 2.0
        channel_freqs = np.full((n_channels, n_bands), centre, dtype=np.float32)
        return channel_freqs, fisher_per_ch, freqs

    freq_res = float(freqs_r[1] - freqs_r[0])
    smooth_bins = max(1, int(round(smooth_hz / freq_res)))
    min_dist_bins = max(1, int(round(bandwidth / freq_res)))

    fisher_smooth = uniform_filter1d(fisher_r, size=smooth_bins, axis=1)

    channel_freqs = np.zeros((n_channels, n_bands), dtype=np.float32)

    for c in range(n_channels):
        curve = fisher_smooth[c]
        peak_idx, _ = find_peaks(curve, distance=min_dist_bins)

        # Sort found peaks by Fisher value (highest first)
        if len(peak_idx) > 0:
            peak_idx = peak_idx[np.argsort(curve[peak_idx])[::-1]]
            selected: List[int] = peak_idx.tolist()
        else:
            selected = []

        # Fill remaining slots greedily with highest-valued non-overlapping bins
        if len(selected) < n_bands:
            for idx in np.argsort(curve)[::-1]:
                if len(selected) >= n_bands:
                    break
                too_close = any(abs(int(idx) - s) < min_dist_bins for s in selected)
                if not too_close:
                    selected.append(int(idx))

        for slot, idx in enumerate(selected[:n_bands]):
            channel_freqs[c, slot] = float(freqs_r[idx])

    # Log a compact summary: mean centre per slot
    slot_means = channel_freqs.mean(axis=0)
    slot_ranges = [
        f"{channel_freqs[:, b].min():.1f}–{channel_freqs[:, b].max():.1f}"
        for b in range(n_bands)
    ]
    logger.info(
        "Channel-specific band selection: %d ch × %d slots  "
        "range=[%.1f, %.1f] Hz  bw=%.1f Hz",
        n_channels, n_bands, lo_hz, hi_hz, bandwidth,
    )
    logger.info(
        "  Per-slot centre ranges (Hz): %s",
        "  ".join(f"slot{b}={slot_ranges[b]}(μ={slot_means[b]:.1f})"
                  for b in range(n_bands)),
    )
    return channel_freqs, fisher_per_ch, freqs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_fisher_curve(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    top_k_channels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate PSD per trial, average over channels, compute Fisher ratio.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        Class labels, shape ``(n_trials,)``.
    sfreq : float
        Sampling frequency.
    nperseg : int
        Welch segment length (clipped to ``n_samples`` if larger).
    top_k_channels : int, optional
        If set, rank channels by their peak Fisher ratio and average PSD
        over only the top-K before computing the Fisher curve.  ``None``
        uses all channels.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis.
    fisher_curve : np.ndarray
        Fisher discriminant ratio per frequency bin.
    """
    n_trials, n_channels, n_samples = X.shape
    nperseg = min(nperseg, n_samples)

    # Flatten trials × channels for a single vectorised Welch call
    X_2d = X.reshape(n_trials * n_channels, n_samples)
    freqs, psd_2d = welch(X_2d, fs=sfreq, nperseg=nperseg, axis=-1)

    psd_3d = psd_2d.reshape(n_trials, n_channels, -1)  # (n_trials, n_ch, n_freqs)

    if top_k_channels is not None and top_k_channels < n_channels:
        # Rank channels by their peak Fisher ratio across frequency bins
        fisher_per_ch = _fisher_ratio_per_channel(psd_3d, y)  # (n_ch, n_freqs)
        ch_scores = fisher_per_ch.max(axis=1)                  # peak Fisher per channel
        top_idx = np.argsort(ch_scores)[::-1][:top_k_channels]
        logger.info(
            "Top-%d channels for band selection (by peak Fisher): %s",
            top_k_channels, top_idx.tolist(),
        )
        psd = psd_3d[:, top_idx, :].mean(axis=1)  # (n_trials, n_freqs)
    else:
        # Default: average over all channels
        psd = psd_3d.mean(axis=1)  # (n_trials, n_freqs)

    fisher_curve = _fisher_ratio(psd, y)

    return freqs, fisher_curve


def _fisher_ratio_per_channel(psd_3d: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fisher discriminant ratio at each (channel, frequency) pair.

    Parameters
    ----------
    psd_3d : np.ndarray
        Shape ``(n_trials, n_channels, n_freqs)``.
    y : np.ndarray
        Class labels, shape ``(n_trials,)``.

    Returns
    -------
    np.ndarray
        Fisher ratio, shape ``(n_channels, n_freqs)``.  Never negative.
    """
    classes = np.unique(y)
    mu_global = psd_3d.mean(axis=0).astype(np.float64)  # (n_ch, n_freqs)

    s_b = np.zeros_like(mu_global, dtype=np.float64)
    s_w = np.zeros_like(mu_global, dtype=np.float64)

    for c in classes:
        mask = y == c
        n_c = int(mask.sum())
        psd_c = psd_3d[mask].astype(np.float64)          # (n_c, n_ch, n_freqs)
        mu_c = psd_c.mean(axis=0)                         # (n_ch, n_freqs)
        s_b += n_c * (mu_c - mu_global) ** 2
        s_w += np.sum((psd_c - mu_c) ** 2, axis=0)

    return s_b / (s_w + 1e-30)


def _fisher_ratio(psd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multi-class Fisher discriminant ratio at each frequency bin.

    Parameters
    ----------
    psd : np.ndarray
        Shape ``(n_trials, n_freqs)``.
    y : np.ndarray
        Class labels, shape ``(n_trials,)``.

    Returns
    -------
    np.ndarray
        Fisher ratio per bin, shape ``(n_freqs,)``.  Never negative.
    """
    classes = np.unique(y)
    mu_global = psd.mean(axis=0)  # (n_freqs,)

    s_b = np.zeros(psd.shape[1], dtype=np.float64)
    s_w = np.zeros(psd.shape[1], dtype=np.float64)

    for c in classes:
        mask = y == c
        n_c = mask.sum()
        psd_c = psd[mask]
        mu_c = psd_c.mean(axis=0)
        s_b += n_c * (mu_c - mu_global) ** 2
        s_w += np.sum((psd_c - mu_c) ** 2, axis=0)

    return s_b / (s_w + 1e-12)


def _generate_candidates(
    bandwidth: float,
    step: float,
    band_range: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Generate all candidate bands with fixed width and step.

    Parameters
    ----------
    bandwidth : float
        Band width in Hz.
    step : float
        Step between band starts in Hz.
    band_range : Tuple[float, float]
        ``(f_min, f_max)`` defining the search range.

    Returns
    -------
    List[Tuple[float, float]]
        All ``(lo, hi)`` candidate bands whose ``hi <= f_max``.
    """
    f_min, f_max = band_range
    candidates = []
    lo = f_min
    while lo + bandwidth <= f_max + 1e-9:
        candidates.append((lo, lo + bandwidth))
        lo += step
    return candidates


def _score_candidates(
    candidates: List[Tuple[float, float]],
    freqs: np.ndarray,
    fisher_curve: np.ndarray,
) -> np.ndarray:
    """Integrate Fisher ratio over each candidate band's passband.

    Parameters
    ----------
    candidates : List[Tuple[float, float]]
        Candidate ``(lo, hi)`` bands.
    freqs : np.ndarray
        Frequency axis from Welch.
    fisher_curve : np.ndarray
        Fisher ratio per bin.

    Returns
    -------
    np.ndarray
        Scalar score per candidate, shape ``(n_candidates,)``.
    """
    scores = np.zeros(len(candidates))
    for i, (lo, hi) in enumerate(candidates):
        mask = (freqs >= lo) & (freqs <= hi)
        scores[i] = fisher_curve[mask].sum() if mask.any() else 0.0
    return scores


def _overlap_fraction(
    band_a: Tuple[float, float],
    band_b: Tuple[float, float],
    bandwidth: float,
) -> float:
    """Fraction of *bandwidth* that two bands overlap.

    Parameters
    ----------
    band_a, band_b : Tuple[float, float]
        ``(lo, hi)`` band boundaries.
    bandwidth : float
        Nominal band width (used for normalisation).

    Returns
    -------
    float
        Overlap in [0, 1].
    """
    intersection = max(0.0, min(band_a[1], band_b[1]) - max(band_a[0], band_b[0]))
    return intersection / bandwidth


def _greedy_select(
    candidates: List[Tuple[float, float]],
    scores: np.ndarray,
    n_bands: int,
    bandwidth: float,
    max_overlap: float = 0.5,
    min_fisher_fraction: float = 0.05,
) -> List[Tuple[float, float]]:
    """Greedy band selection with overlap and Fisher score threshold constraints.

    Sort candidates by descending score, add each if:
      (a) its maximum pairwise overlap with already-selected bands ≤ max_overlap, and
      (b) its score ≥ top_score * min_fisher_fraction.

    Constraint (b) prevents low-discriminability noise bands from being forced in
    when n_bands is larger than the number of genuinely informative bands.

    Parameters
    ----------
    candidates : List[Tuple[float, float]]
        All candidate bands.
    scores : np.ndarray
        Score per candidate.
    n_bands : int
        Maximum number of bands to select.
    bandwidth : float
        Nominal band width for overlap normalisation.
    max_overlap : float
        Maximum allowed fractional overlap with any selected band.
    min_fisher_fraction : float
        Minimum score as a fraction of the top candidate score.

    Returns
    -------
    List[Tuple[float, float]]
        Selected bands, ordered by score (highest first).
    """
    order = np.argsort(scores)[::-1]
    score_threshold = scores[order[0]] * min_fisher_fraction
    selected: List[Tuple[float, float]] = []

    for idx in order:
        if scores[idx] < score_threshold:
            logger.info(
                "Band selection: stopping early — score %.4f below threshold %.4f (%.0f%% of top)",
                scores[idx], score_threshold, min_fisher_fraction * 100,
            )
            break
        candidate = candidates[idx]
        if all(
            _overlap_fraction(candidate, sel, bandwidth) <= max_overlap
            for sel in selected
        ):
            selected.append(candidate)
        if len(selected) >= n_bands:
            break

    if len(selected) < n_bands:
        logger.warning(
            "Selected %d bands (requested %d) — remaining candidates below Fisher threshold.",
            len(selected), n_bands,
        )

    return selected
