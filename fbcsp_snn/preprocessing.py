"""Bandpass filter bank and Pairwise CSP with dual-end eigenvector extraction.

Filter bank
-----------
Applies a zero-phase Butterworth bandpass filter for each frequency band and
returns a list of per-band arrays so that downstream code can access each
band independently (needed for per-band CSP fitting).

Pairwise CSP
------------
For every ordered pair of distinct classes ``(c1, c2)`` (all C*(C-1)/2 unique
pairs):

1. Compute the average normalised covariance matrix for each class.
2. Regularise: ``Σ_reg = (1 - λ) Σ + λ I``.
3. Solve the generalised eigenvalue problem
   ``Σ_A W = λ_eig (Σ_A + Σ_B) W``.
4. Take the first *m* **and** last *m* eigenvectors (dual-end extraction):
   - first *m* → maximise class-B variance / minimise class-A variance
   - last *m*  → maximise class-A variance / minimise class-B variance
5. Project each band's filtered data through the pair's spatial filters.

For 4 classes, 6 pairs, 6 bands and m=2 this yields
``6 × 4 = 24`` features per band → ``6 × 24 = 144`` features total (stored as
time series of length ``n_samples``).

All fitting must be done **only on training data**; ``transform`` applies the
training-derived filters to any split.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

from fbcsp_snn import setup_logger

logger: logging.Logger = setup_logger(__name__)

# Type alias: class pair key
_Pair = Tuple[int, int]


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    X: np.ndarray,
    lo: float,
    hi: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to EEG data.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape ``(n_trials, n_channels, n_samples)``.
    lo : float
        Lower cutoff frequency in Hz.
    hi : float
        Upper cutoff frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered data, same shape as *X*, dtype ``float32``.
    """
    nyq = sfreq / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    # Clamp to valid range to avoid Butterworth instability near 0 or Nyquist
    lo_n = np.clip(lo_n, 1e-4, 1.0 - 1e-4)
    hi_n = np.clip(hi_n, 1e-4, 1.0 - 1e-4)

    sos = butter(order, [lo_n, hi_n], btype="bandpass", output="sos")

    n_trials, n_channels, n_samples = X.shape
    # Reshape to (n_trials * n_channels, n_samples) for a single vectorised call
    X_2d = X.reshape(n_trials * n_channels, n_samples)
    X_filt_2d = sosfiltfilt(sos, X_2d, axis=-1)
    return X_filt_2d.reshape(n_trials, n_channels, n_samples).astype(np.float32)


def apply_filter_bank(
    X: np.ndarray,
    bands: List[Tuple[float, float]],
    sfreq: float,
    order: int = 4,
) -> List[np.ndarray]:
    """Apply a bank of bandpass filters to EEG data.

    Parameters
    ----------
    X : np.ndarray
        Raw EEG, shape ``(n_trials, n_channels, n_samples)``.
    bands : List[Tuple[float, float]]
        List of ``(lo, hi)`` frequency bands in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    List[np.ndarray]
        One filtered array per band, each shape
        ``(n_trials, n_channels, n_samples)``.  The concatenation along the
        channel axis yields shape ``(n_trials, n_channels × n_bands, n_samples)``
        as documented in the pipeline architecture.
    """
    n_trials, n_channels, n_samples = X.shape
    filtered: List[np.ndarray] = []

    for lo, hi in bands:
        X_band = bandpass_filter(X, lo, hi, sfreq, order=order)
        filtered.append(X_band)
        logger.debug("Filtered band (%.1f–%.1f Hz): %s", lo, hi, X_band.shape)

    logger.info(
        "Filter bank: %d bands → each %s  (concat would be %s)",
        len(bands),
        (n_trials, n_channels, n_samples),
        (n_trials, n_channels * len(bands), n_samples),
    )
    return filtered


# ---------------------------------------------------------------------------
# Pairwise CSP
# ---------------------------------------------------------------------------

class PairwiseCSP:
    """Pairwise CSP with dual-end eigenvector extraction.

    Parameters
    ----------
    m : int
        Number of eigenvectors taken from each end of the spectrum per pair.
        Total spatial filters per (band, pair) = ``2 * m``.
    lambda_r : float
        Tikhonov regularisation strength applied to each covariance matrix.
        ``Σ_reg = (1 - λ) Σ + λ I``.

    Attributes
    ----------
    filters_ : Dict[Tuple[int, _Pair], np.ndarray]
        Fitted spatial filters.  Key is ``(band_idx, (c1, c2))``, value is
        ``W`` of shape ``(n_channels, 2 * m)``.
    pairs_ : List[_Pair]
        All unique class pairs, sorted.
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    """

    def __init__(self, m: int = 2, lambda_r: float = 0.0001) -> None:
        self.m = m
        self.lambda_r = lambda_r
        self.filters_: Dict[Tuple[int, _Pair], np.ndarray] = {}
        self.pairs_: List[_Pair] = []
        self.classes_: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_bands: List[np.ndarray],
        y: np.ndarray,
    ) -> "PairwiseCSP":
        """Fit spatial filters on training data.

        Parameters
        ----------
        X_bands : List[np.ndarray]
            Per-band filtered EEG, each ``(n_trials, n_channels, n_samples)``.
        y : np.ndarray
            Class labels, shape ``(n_trials,)``.  Must be 1-indexed integers.

        Returns
        -------
        PairwiseCSP
            Self (for chaining).
        """
        self.classes_ = np.unique(y)
        self.pairs_ = sorted(combinations(self.classes_, 2))  # type: ignore[arg-type]
        self.filters_ = {}

        n_channels = X_bands[0].shape[1]

        for b_idx, X_band in enumerate(X_bands):
            for pair in self.pairs_:
                c1, c2 = pair
                cov_a = _mean_normalised_cov(X_band[y == c1])
                cov_b = _mean_normalised_cov(X_band[y == c2])
                cov_a_reg = _regularise(cov_a, self.lambda_r)
                cov_b_reg = _regularise(cov_b, self.lambda_r)

                W = _solve_csp(cov_a_reg, cov_b_reg, self.m)
                self.filters_[(b_idx, pair)] = W

        n_bands = len(X_bands)
        n_pairs = len(self.pairs_)
        logger.info(
            "PairwiseCSP fit: %d bands × %d pairs × %d filters = %d total filters",
            n_bands, n_pairs, 2 * self.m, n_bands * n_pairs * 2 * self.m,
        )
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(
        self,
        X_bands: List[np.ndarray],
    ) -> Dict[_Pair, np.ndarray]:
        """Project filtered data through fitted spatial filters.

        For each class pair the projections from all bands are concatenated
        along the feature axis.

        Parameters
        ----------
        X_bands : List[np.ndarray]
            Per-band filtered EEG, each ``(n_trials, n_channels, n_samples)``.

        Returns
        -------
        Dict[_Pair, np.ndarray]
            Maps ``(c1, c2)`` → projected array of shape
            ``(n_trials, 2 * m * n_bands, n_samples)``.
        """
        if not self.filters_:
            raise RuntimeError("Call fit() before transform().")

        projections: Dict[_Pair, List[np.ndarray]] = {p: [] for p in self.pairs_}

        for b_idx, X_band in enumerate(X_bands):
            for pair in self.pairs_:
                W = self.filters_[(b_idx, pair)]  # (n_channels, 2m)
                # W.T: (2m, n_channels) × X_band.T-like → use einsum
                # X_band: (n_trials, n_channels, n_samples)
                # projected: (n_trials, 2m, n_samples)
                # W: (n_channels, 2m)  X_band: (n_trials, n_channels, n_samples)
                # -> proj: (n_trials, 2m, n_samples)
                proj = np.einsum("cf,tcs->tfs", W, X_band)
                projections[pair].append(proj)

        result: Dict[_Pair, np.ndarray] = {}
        for pair in self.pairs_:
            # Concatenate bands along feature axis → (n_trials, 2m*n_bands, n_samples)
            result[pair] = np.concatenate(projections[pair], axis=1)

        return result

    def fit_transform(
        self,
        X_bands: List[np.ndarray],
        y: np.ndarray,
    ) -> Dict[_Pair, np.ndarray]:
        """Fit and immediately transform the same data.

        Parameters
        ----------
        X_bands : List[np.ndarray]
            Per-band filtered EEG.
        y : np.ndarray
            Class labels.

        Returns
        -------
        Dict[_Pair, np.ndarray]
            Same output as :meth:`transform`.
        """
        return self.fit(X_bands, y).transform(X_bands)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mean_normalised_cov(X_trials: np.ndarray) -> np.ndarray:
    """Compute the mean normalised covariance matrix over a set of trials.

    Each trial's covariance is normalised by its trace before averaging so
    that differences in overall signal amplitude do not dominate.

    Parameters
    ----------
    X_trials : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.

    Returns
    -------
    np.ndarray
        Averaged normalised covariance, shape ``(n_channels, n_channels)``.
    """
    n_trials, n_channels, n_samples = X_trials.shape
    # Batch outer product: (n_trials, n_channels, n_channels)
    covs = np.einsum("tcs,tds->tcd", X_trials, X_trials)
    # Normalise each trial by its trace
    traces = np.trace(covs, axis1=1, axis2=2)[:, None, None]  # (n_trials, 1, 1)
    covs_norm = covs / (traces + 1e-12)
    return covs_norm.mean(axis=0)


def _regularise(cov: np.ndarray, lambda_r: float) -> np.ndarray:
    """Apply Tikhonov regularisation: ``(1 - λ) Σ + λ I``.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix, shape ``(n_channels, n_channels)``.
    lambda_r : float
        Regularisation coefficient in ``[0, 1)``.

    Returns
    -------
    np.ndarray
        Regularised covariance, same shape.
    """
    n = cov.shape[0]
    return (1.0 - lambda_r) * cov + lambda_r * np.eye(n)


def _solve_csp(
    cov_a: np.ndarray,
    cov_b: np.ndarray,
    m: int,
) -> np.ndarray:
    """Solve the generalised eigenvalue problem and return dual-end filters.

    Solves ``Σ_A W = λ (Σ_A + Σ_B) W`` using ``scipy.linalg.eigh``
    (symmetric solver, eigenvalues in ascending order).

    Parameters
    ----------
    cov_a : np.ndarray
        Regularised covariance for class A, shape ``(n_channels, n_channels)``.
    cov_b : np.ndarray
        Regularised covariance for class B, shape ``(n_channels, n_channels)``.
    m : int
        Number of filters from each end.

    Returns
    -------
    np.ndarray
        Spatial filter matrix ``W``, shape ``(n_channels, 2 * m)``.
        Columns ``0 … m-1`` are the first *m* eigenvectors (low eigenvalue →
        maximise class-B variance); columns ``m … 2m-1`` are the last *m*
        eigenvectors (high eigenvalue → maximise class-A variance).
    """
    composite = cov_a + cov_b
    # eigh returns eigenvalues ascending, eigenvectors as columns
    _, W = eigh(cov_a, composite)
    return np.concatenate([W[:, :m], W[:, -m:]], axis=1)


# ---------------------------------------------------------------------------
# Z-normalisation
# ---------------------------------------------------------------------------

class ZNormaliser:
    """Per-feature z-normalisation of CSP projection arrays.

    Fitted on the concatenated training projection
    ``(n_trials, n_features, n_samples)``; applied identically to val/test
    splits.  Statistics are computed across the (n_trials × n_samples)
    dimension so that each spatial-filter output channel has zero mean and
    unit variance.

    Attributes
    ----------
    mean_ : np.ndarray
        Per-feature mean, shape ``(n_features,)``.
    std_ : np.ndarray
        Per-feature standard deviation (+ 1e-8), shape ``(n_features,)``.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "ZNormaliser":
        """Compute per-feature mean and std from training projections.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_trials, n_features, n_samples)``.

        Returns
        -------
        ZNormaliser
            Self (for chaining).
        """
        # Reshape to (n_features, n_trials * n_samples)
        n_trials, n_features, n_samples = X.shape
        X_2d = X.transpose(1, 0, 2).reshape(n_features, n_trials * n_samples)
        self.mean_ = X_2d.mean(axis=1)            # (n_features,)
        self.std_  = X_2d.std(axis=1) + 1e-8     # (n_features,)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalisation.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_trials, n_features, n_samples)``.

        Returns
        -------
        np.ndarray
            Normalised array, same shape, dtype ``float32``.
        """
        return ((X - self.mean_[None, :, None])
                / self.std_[None, :, None]).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and immediately transform.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_trials, n_features, n_samples)``.

        Returns
        -------
        np.ndarray
            Normalised array, same shape, dtype ``float32``.
        """
        return self.fit(X).transform(X)
