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
from typing import List, Tuple

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
    """
    fisher_freqs, fisher_curve = _compute_fisher_curve(X, y, sfreq)

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
    """
    fisher_freqs, fisher_curve = _compute_fisher_curve(X, y, sfreq)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_fisher_curve(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
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

    # Average PSD over channels → (n_trials, n_freqs)
    psd = psd_2d.reshape(n_trials, n_channels, -1).mean(axis=1)

    fisher_curve = _fisher_ratio(psd, y)

    return freqs, fisher_curve


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
