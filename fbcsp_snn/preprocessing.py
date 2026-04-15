"""Bandpass filter bank and Pairwise CSP with dual-end eigenvector extraction.

Filter bank
-----------
Applies a causal (single forward-pass) Butterworth bandpass filter for each
frequency band and returns a list of per-band arrays so that downstream code
can access each band independently (needed for per-band CSP fitting).
Causal filtering is used so that the filter bank maps directly to an analog
Gm-C circuit implementation on neuromorphic hardware.

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
from scipy.signal import butter, bessel, sosfilt
from sklearn.covariance import ledoit_wolf as _lw_estimate

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
    filter_type: str = "butterworth",
) -> np.ndarray:
    """Apply a causal bandpass filter to EEG data.

    Uses a single forward pass (``sosfilt``) so the filter maps to a
    real-time analog Gm-C circuit for neuromorphic deployment.

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
        Filter order.
    filter_type : str
        ``'butterworth'`` — maximally flat magnitude, moderate group delay.
        ``'bessel'``      — maximally flat group delay, minimal phase distortion.

    Returns
    -------
    np.ndarray
        Filtered data, same shape as *X*, dtype ``float32``.
    """
    nyq = sfreq / 2.0
    lo_n = np.clip(lo / nyq, 1e-4, 1.0 - 1e-4)
    hi_n = np.clip(hi / nyq, 1e-4, 1.0 - 1e-4)

    if filter_type == "bessel":
        # norm='delay' → maximally flat group delay (minimal timing distortion)
        sos = bessel(order, [lo_n, hi_n], btype="bandpass", norm="delay", output="sos")
    else:
        sos = butter(order, [lo_n, hi_n], btype="bandpass", output="sos")

    n_trials, n_channels, n_samples = X.shape
    X_2d = X.reshape(n_trials * n_channels, n_samples)
    X_filt_2d = sosfilt(sos, X_2d, axis=-1)
    return X_filt_2d.reshape(n_trials, n_channels, n_samples).astype(np.float32)


def window_filter_bank(
    X_bands: List[np.ndarray],
    y: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Slide overlapping windows over filter-bank output for augmented CSP fitting.

    Used exclusively to augment the training set before CSP covariance
    estimation.  Val and test data are never windowed — CSP spatial filters
    are still applied to full-length trials downstream.

    Parameters
    ----------
    X_bands : List[np.ndarray]
        Per-band filtered EEG, each ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        Class labels, shape ``(n_trials,)``.
    window_samples : int
        Window length in samples.
    step_samples : int
        Step between consecutive window starts in samples.

    Returns
    -------
    List[np.ndarray]
        Windowed data per band, each
        ``(n_trials × n_windows_per_trial, n_channels, window_samples)``.
    np.ndarray
        Replicated labels, shape ``(n_trials × n_windows_per_trial,)``.
    """
    n_trials, _, n_samples = X_bands[0].shape
    starts = list(range(0, n_samples - window_samples + 1, step_samples))
    n_win = len(starts)

    y_aug = np.repeat(y, n_win)

    X_bands_aug: List[np.ndarray] = []
    for X_band in X_bands:
        # (n_trials, n_channels, n_samples) → (n_trials*n_win, n_channels, window_samples)
        windows = np.stack(
            [X_band[:, :, s:s + window_samples] for s in starts],
            axis=1,                              # (n_trials, n_win, n_channels, window_samples)
        ).reshape(n_trials * n_win, X_band.shape[1], window_samples)
        X_bands_aug.append(windows)

    return X_bands_aug, y_aug


def apply_filter_bank(
    X: np.ndarray,
    bands: List[Tuple[float, float]],
    sfreq: float,
    order: int = 4,
    filter_type: str = "butterworth",
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
        Filter order.
    filter_type : str
        ``'butterworth'`` or ``'bessel'``.

    Returns
    -------
    List[np.ndarray]
        One filtered array per band, each shape
        ``(n_trials, n_channels, n_samples)``.
    """
    filtered: List[np.ndarray] = []

    for lo, hi in bands:
        X_band = bandpass_filter(X, lo, hi, sfreq, order=order,
                                 filter_type=filter_type)
        filtered.append(X_band)
        logger.debug("Filtered band (%.1f–%.1f Hz): %s", lo, hi, X_band.shape)

    n_trials, n_channels, n_samples = X.shape
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
    """Pairwise CSP with dual-end eigenvector extraction and optional EA.

    Euclidean Alignment (EA, He et al. 2019) is applied per band before CSP
    fitting.  For each band it computes the mean covariance ``R`` of all
    training trials (class-agnostic) and whitens every trial by ``R^{-1/2}``.
    The whiteners are stored and reused in ``transform`` so val/test data are
    aligned with the training distribution without leakage.

    Parameters
    ----------
    m : int
        Number of eigenvectors taken from each end of the spectrum per pair.
        Total spatial filters per (band, pair) = ``2 * m``.
    lambda_r : float
        Tikhonov regularisation strength applied to each covariance matrix.
        ``Σ_reg = (1 - λ) Σ + λ I``.
    euclidean_alignment : bool
        If ``True`` (default), apply Euclidean Alignment before CSP fitting.
    riemannian_mean : bool
        If ``True`` (default), use the Riemannian (Fréchet) mean of the
        per-class covariance matrices instead of the arithmetic mean.
        Avoids the SPD swelling effect and better represents the geometric
        centre of the covariance distribution on the manifold.

    Attributes
    ----------
    filters_ : Dict[Tuple[int, _Pair], np.ndarray]
        Fitted spatial filters.  Key is ``(band_idx, (c1, c2))``, value is
        ``W`` of shape ``(n_channels, 2 * m)``.
    ea_whiteners_ : Dict[int, np.ndarray]
        Per-band EA whitener ``R^{-1/2}``, shape ``(n_channels, n_channels)``.
        Empty when ``euclidean_alignment=False``.
    pairs_ : List[_Pair]
        All unique class pairs, sorted.
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    """

    def __init__(
        self,
        m: int = 2,
        lambda_r: float = 0.0001,
        euclidean_alignment: bool = True,
        riemannian_mean: bool = True,
        ledoit_wolf: bool = False,
    ) -> None:
        self.m = m
        self.lambda_r = lambda_r
        self.euclidean_alignment = euclidean_alignment
        self.riemannian_mean = riemannian_mean
        self.ledoit_wolf = ledoit_wolf
        self.filters_: Dict[Tuple[int, _Pair], np.ndarray] = {}
        self.ea_whiteners_: Dict[int, np.ndarray] = {}
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
            # ---- Euclidean Alignment ----
            if self.euclidean_alignment:
                R_invsqrt = _compute_ea_whitener(X_band)
                self.ea_whiteners_[b_idx] = R_invsqrt
                X_band = _apply_ea(X_band, R_invsqrt)

            for pair in self.pairs_:
                c1, c2 = pair
                if self.ledoit_wolf:
                    # LW per-trial covariances → Riemannian mean (or arithmetic).
                    # Replaces only the regularisation; averaging method follows
                    # the riemannian_mean flag so the comparison vs V4.1 is clean.
                    covs_a = _ledoit_wolf_covs(X_band[y == c1])
                    covs_b = _ledoit_wolf_covs(X_band[y == c2])
                    if self.riemannian_mean:
                        cov_a = _riemannian_mean_from_covs(covs_a)
                        cov_b = _riemannian_mean_from_covs(covs_b)
                    else:
                        cov_a = covs_a.mean(axis=0)
                        cov_b = covs_b.mean(axis=0)
                    W = _solve_csp(cov_a, cov_b, self.m)
                else:
                    _cov_fn = _riemannian_mean_cov if self.riemannian_mean else _mean_normalised_cov
                    cov_a = _cov_fn(X_band[y == c1])
                    cov_b = _cov_fn(X_band[y == c2])
                    cov_a = _regularise(cov_a, self.lambda_r)
                    cov_b = _regularise(cov_b, self.lambda_r)
                    W = _solve_csp(cov_a, cov_b, self.m)
                self.filters_[(b_idx, pair)] = W

        n_bands = len(X_bands)
        n_pairs = len(self.pairs_)
        logger.info(
            "PairwiseCSP fit: %d bands × %d pairs × %d filters = %d total filters"
            "  (EA: %s  RiemannMean: %s  LedoitWolf: %s)",
            n_bands, n_pairs, 2 * self.m, n_bands * n_pairs * 2 * self.m,
            self.euclidean_alignment, self.riemannian_mean, self.ledoit_wolf,
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
            # Apply stored EA whitener (fitted on training data)
            if self.euclidean_alignment and b_idx in self.ea_whiteners_:
                X_band = _apply_ea(X_band, self.ea_whiteners_[b_idx])

            for pair in self.pairs_:
                W = self.filters_[(b_idx, pair)]  # (n_channels, 2m)
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

def _compute_ea_whitener(X: np.ndarray) -> np.ndarray:
    """Compute the EA whitener ``R^{-1/2}`` from a set of trials.

    ``R`` is the arithmetic mean of per-trial covariance matrices
    (class-agnostic).  The inverse square root is computed via eigendecomposition
    of the symmetric matrix for numerical stability.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.

    Returns
    -------
    np.ndarray
        ``R^{-1/2}``, shape ``(n_channels, n_channels)``.
    """
    n_trials, n_channels, n_samples = X.shape
    # Per-trial covariance (n_trials, n_channels, n_channels)
    covs = np.einsum("tcs,tds->tcd", X, X) / n_samples
    R = covs.mean(axis=0)  # (n_channels, n_channels)
    # Eigendecomposition (eigh: symmetric, eigenvalues ascending)
    eigenvalues, eigenvectors = eigh(R)
    eigenvalues = np.maximum(eigenvalues, 1e-10)   # clamp numerical negatives
    R_invsqrt = (eigenvectors * (eigenvalues ** -0.5)[None, :]) @ eigenvectors.T
    return R_invsqrt.astype(np.float64)


def _apply_ea(X: np.ndarray, R_invsqrt: np.ndarray) -> np.ndarray:
    """Apply EA whitening: ``X̃_i = R^{-1/2} @ X_i`` for each trial.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    R_invsqrt : np.ndarray
        Shape ``(n_channels, n_channels)``.

    Returns
    -------
    np.ndarray
        Whitened array, same shape as *X*.
    """
    return np.einsum("cd,tds->tcs", R_invsqrt, X).astype(np.float32)


def _spd_sqrt_invsqrt(M: np.ndarray) -> tuple:
    """Compute (M^{1/2}, M^{-1/2}) for an SPD matrix via eigendecomposition.

    Parameters
    ----------
    M : np.ndarray
        Symmetric positive definite matrix, shape ``(n, n)``.

    Returns
    -------
    tuple
        ``(M_sqrt, M_invsqrt)``, each shape ``(n, n)``.
    """
    vals, vecs = eigh(M)
    vals = np.maximum(vals, 1e-10)
    M_sqrt    = (vecs * np.sqrt(vals))        @ vecs.T
    M_invsqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    return M_sqrt, M_invsqrt


def _spd_log(S: np.ndarray) -> np.ndarray:
    """Matrix logarithm of an SPD matrix S = V diag(d) V^T → V diag(log d) V^T."""
    vals, vecs = eigh(S)
    vals = np.maximum(vals, 1e-10)
    return (vecs * np.log(vals)) @ vecs.T


def _spd_exp(S: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix S = V diag(d) V^T → V diag(exp d) V^T."""
    vals, vecs = eigh(S)
    return (vecs * np.exp(vals)) @ vecs.T


def _riemannian_mean_from_covs(
    covs: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    """Riemannian (Fréchet) mean of a stack of SPD matrices.

    Uses gradient descent on the SPD manifold (Moakher 2005).  Accepts
    pre-computed covariance matrices so that different covariance estimators
    (e.g. Ledoit-Wolf) can feed into the same Riemannian averaging step.

    Parameters
    ----------
    covs : np.ndarray
        Pre-computed, trace-normalised SPD matrices,
        shape ``(n_trials, n_channels, n_channels)``.
    max_iter : int
        Maximum gradient-descent iterations.
    tol : float
        Frobenius-norm convergence threshold.

    Returns
    -------
    np.ndarray
        Riemannian mean, shape ``(n_channels, n_channels)``.
    """
    n_trials = covs.shape[0]
    M = covs.mean(axis=0)   # arithmetic initialisation

    for _ in range(max_iter):
        M_sqrt, M_invsqrt = _spd_sqrt_invsqrt(M)

        grad = np.zeros_like(M)
        for C in covs:
            grad += _spd_log(M_invsqrt @ C @ M_invsqrt)
        grad /= n_trials

        M_new = M_sqrt @ _spd_exp(grad) @ M_sqrt

        if np.linalg.norm(M_new - M, "fro") < tol:
            M = M_new
            break
        M = M_new

    return M


def _riemannian_mean_cov(
    X_trials: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    """Riemannian mean of trace-normalised covariances computed from raw trials.

    Thin wrapper around :func:`_riemannian_mean_from_covs` that handles the
    raw-data → covariance step internally.

    Parameters
    ----------
    X_trials : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    max_iter : int
        Maximum gradient-descent iterations.
    tol : float
        Frobenius-norm convergence threshold.

    Returns
    -------
    np.ndarray
        Riemannian mean covariance, shape ``(n_channels, n_channels)``.
    """
    covs = np.einsum("tcs,tds->tcd", X_trials, X_trials)   # (n_trials, C, C)
    traces = np.trace(covs, axis1=1, axis2=2)[:, None, None]
    covs = covs / (traces + 1e-12)
    return _riemannian_mean_from_covs(covs, max_iter=max_iter, tol=tol)


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


def _ledoit_wolf_covs(X_trials: np.ndarray) -> np.ndarray:
    """Per-trial Ledoit-Wolf regularised, trace-normalised covariance matrices.

    Each trial is regularised with analytically optimal shrinkage
    (Ledoit & Wolf 2004) and trace-normalised.  Returns the full stack of
    per-trial SPD matrices so the caller can pass them to either
    :func:`_riemannian_mean_from_covs` or take an arithmetic mean.

    Parameters
    ----------
    X_trials : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.

    Returns
    -------
    np.ndarray
        Per-trial regularised covariances, shape
        ``(n_trials, n_channels, n_channels)``.
    """
    n_trials, n_channels, _ = X_trials.shape
    covs = np.empty((n_trials, n_channels, n_channels), dtype=np.float64)
    for i, trial in enumerate(X_trials):    # trial: (n_channels, n_samples)
        cov, _ = _lw_estimate(trial.T)      # sklearn expects (n_samples, n_channels)
        tr = np.trace(cov)
        covs[i] = cov / (tr + 1e-12)
    return covs


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
