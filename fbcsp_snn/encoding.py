"""Adaptive-threshold spike encoding with JIT-compiled inner loop.

Algorithm (delta-based)
-----------------------
For each timestep *t* and feature *f*:

1. Compute the absolute signal delta: ``|x[t,f] - x[t-1,f]|``
2. If delta > threshold[f]: emit a spike (1), else 0
3. After each step: ``threshold *= decay``
4. On spike: ``threshold += adapt_inc``

The adaptive threshold rises when the signal is changing rapidly and recovers
exponentially, producing a sparse, rate-coded spike representation.

The per-timestep loop is JIT-compiled via ``@torch.jit.script`` so the Python
interpreter overhead disappears; inner operations are vectorised across the
batch and feature dimensions.

Input  → ``encode_csp_projections``:  dict of CSP projections
Output → binary spike tensor ``(n_timesteps, n_trials, n_features)``
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger

logger: logging.Logger = setup_logger(__name__)

# Type alias shared with preprocessing
_Pair = Tuple[int, int]


# ---------------------------------------------------------------------------
# JIT-compiled encoder kernel
# ---------------------------------------------------------------------------

@torch.jit.script
def _adaptive_threshold_encode_jit(
    x: torch.Tensor,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    """JIT-compiled adaptive threshold spike encoder.

    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape ``(T, batch, n_features)``.  Must be float.
    base_thresh : float
        Initial threshold value applied uniformly to every feature.
    adapt_inc : float
        Amount added to the threshold whenever a spike is emitted.
    decay : float
        Multiplicative decay applied to the threshold at every timestep.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, same shape as *x*.  dtype matches *x*.
    """
    T: int = x.shape[0]
    batch: int = x.shape[1]
    n_feat: int = x.shape[2]

    spikes = torch.zeros_like(x)
    threshold = torch.full(
        (batch, n_feat),
        base_thresh,
        dtype=x.dtype,
        device=x.device,
    )

    for t in range(1, T):
        delta = torch.abs(x[t] - x[t - 1])          # (batch, n_feat)
        fired = (delta > threshold).to(x.dtype)       # (batch, n_feat)
        spikes[t] = fired
        # Decay threshold every step; increment on spike
        threshold = threshold * decay + fired * adapt_inc

    return spikes


@torch.jit.script
def _adm_encode_jit(
    x: torch.Tensor,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    """JIT-compiled Asynchronous Delta Modulation (ADM) spike encoder.

    Tracks the signal with a reference voltage ``v_ref``.  Emits an ON spike
    when the signal rises above ``v_ref + threshold``, and an OFF spike when
    it falls below ``v_ref - threshold``.  ``v_ref`` steps toward the signal
    on every spike.  The threshold adapts identically to the delta encoder.

    Doubles the feature dimension by concatenating ON and OFF spike channels.

    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape ``(T, batch, n_features)``.
    base_thresh, adapt_inc, decay : float
        Encoding hyperparameters (shared with delta encoder).

    Returns
    -------
    torch.Tensor
        Binary spike tensor, shape ``(T, batch, 2 * n_features)``.
    """
    T: int = x.shape[0]
    batch: int = x.shape[1]
    n_feat: int = x.shape[2]

    on_spikes  = torch.zeros_like(x)
    off_spikes = torch.zeros_like(x)
    v_ref      = x[0].clone()
    threshold  = torch.full(
        (batch, n_feat), base_thresh, dtype=x.dtype, device=x.device,
    )

    for t in range(T):
        diff     = x[t] - v_ref
        on_fire  = (diff >  threshold).to(x.dtype)
        off_fire = (diff < -threshold).to(x.dtype)
        on_spikes[t]  = on_fire
        off_spikes[t] = off_fire
        v_ref     = v_ref + on_fire * threshold - off_fire * threshold
        threshold = threshold * decay + (on_fire + off_fire) * adapt_inc

    return torch.cat([on_spikes, off_spikes], dim=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_csp_projections(
    projections: Dict[_Pair, np.ndarray],
    base_thresh: float = 0.001,
    adapt_inc: float = 0.6,
    decay: float = 0.95,
    device: Optional[torch.device] = None,
    encoder_type: str = "delta",
) -> torch.Tensor:
    """Encode a dict of CSP-projected time-series into binary spike trains.

    All class-pair projections are concatenated along the feature axis before
    encoding so that the single returned tensor covers the full feature set.
    Pair ordering is alphabetically sorted for reproducibility.

    Parameters
    ----------
    projections : Dict[Tuple[int, int], np.ndarray]
        Output of :meth:`PairwiseCSP.transform`.  Maps each class pair to an
        array of shape ``(n_trials, n_features_per_pair, n_samples)``.
    base_thresh : float
        Initial adaptive threshold.
    adapt_inc : float
        Threshold increment per emitted spike.
    decay : float
        Per-timestep threshold decay factor (< 1).
    device : Optional[torch.device]
        Target device.  Defaults to :data:`fbcsp_snn.DEVICE`.

    Returns
    -------
    torch.Tensor
        Binary spike tensor, shape ``(n_samples, n_trials, total_features)``,
        dtype ``torch.float32``, on *device*.
    """
    if device is None:
        device = DEVICE

    # Sort pairs for deterministic feature ordering
    pairs = sorted(projections.keys())
    arrays = [projections[p] for p in pairs]

    # Concatenate along feature dim → (n_trials, total_features, n_samples)
    X = np.concatenate(arrays, axis=1).astype(np.float32)

    # Move to device and permute → (n_samples, n_trials, total_features) = (T, B, F)
    X_t = torch.from_numpy(X).to(device).permute(2, 0, 1)

    spikes = encode_tensor(X_t, base_thresh, adapt_inc, decay, encoder_type)

    firing_rate = spikes.mean().item()
    logger.info(
        "Spike encoding complete — input: %s -> spikes: %s  mean firing rate: %.4f",
        tuple(X_t.shape),
        tuple(spikes.shape),
        firing_rate,
    )
    return spikes


def encode_tensor(
    X: torch.Tensor,
    base_thresh: float = 0.001,
    adapt_inc: float = 0.6,
    decay: float = 0.95,
    encoder_type: str = "delta",
) -> torch.Tensor:
    """Encode a pre-arranged tensor ``(T, batch, features)`` to spikes.

    Convenience wrapper for use when data is already a tensor in the correct
    layout (e.g. inside the training loop after z-normalisation).

    Parameters
    ----------
    X : torch.Tensor
        Input signal ``(T, batch, n_features)``.
    base_thresh, adapt_inc, decay : float
        Encoding hyperparameters.
    encoder_type : str
        ``"delta"`` — adaptive-threshold delta encoder (default).
        ``"adm"``   — Asynchronous Delta Modulation; doubles feature dimension
        by emitting separate ON and OFF spike channels.

    Returns
    -------
    torch.Tensor
        Binary spikes.  Shape matches *X* for ``"delta"``;
        ``(T, batch, 2 * n_features)`` for ``"adm"``.
    """
    if encoder_type == "adm":
        return _adm_encode_jit(X, base_thresh, adapt_inc, decay)
    return _adaptive_threshold_encode_jit(X, base_thresh, adapt_inc, decay)
