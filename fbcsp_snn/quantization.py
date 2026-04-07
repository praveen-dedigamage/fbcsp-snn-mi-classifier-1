"""Simulated INT8 symmetric per-tensor quantization.

Neuromorphic hardware typically operates with low-precision fixed-point
arithmetic.  This module simulates INT8 by:

1. Computing a per-tensor scale factor:  ``scale = max(|W|) / 127``
2. Quantising weights:  ``Wq = round(W / scale).clamp(-127, 127)``
3. Dequantising back to float:  ``Wd = Wq * scale``

The dequantised tensor has the same dtype and shape as the input but can only
represent ``255`` distinct values (``-127 … 127``), so it accumulates the
rounding error that INT8 inference would introduce.

Both model ``Linear`` weights and numpy CSP filter matrices are supported.
Biases are kept at full precision (standard practice — INT8 biases add
hardware complexity for minimal benefit).

Note
----
This is *simulated* quantisation for evaluation purposes only.  It does not
produce actual ``torch.int8`` tensors and cannot be run on neuromorphic
hardware directly; the point is to measure accuracy degradation from weight
quantisation before committing to a hardware port.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from fbcsp_snn import setup_logger
from fbcsp_snn.model import SNNClassifier

logger: logging.Logger = setup_logger(__name__)

_INT8_MAX: int = 127   # symmetric INT8 range: [-127, 127]


# ---------------------------------------------------------------------------
# Core quant/dequant
# ---------------------------------------------------------------------------

def quantize_tensor_symmetric(
    x: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, float]:
    """Simulate symmetric per-tensor quantisation and return dequantised result.

    Parameters
    ----------
    x : torch.Tensor
        Floating-point weight tensor (any shape).
    bits : int
        Bit-width.  Default ``8`` → range ``[-127, 127]``.

    Returns
    -------
    dequantised : torch.Tensor
        Tensor with the same shape and dtype as *x* but rounded to the nearest
        representable INT*bits* value.
    scale : float
        Quantisation scale (``max(|x|) / q_max``).  Zero if *x* is all-zero.
    """
    q_max = float(2 ** (bits - 1) - 1)   # 127 for INT8
    abs_max = x.abs().max().item()
    if abs_max == 0.0:
        return x.clone(), 1.0

    scale = abs_max / q_max
    q = x.div(scale).round().clamp(-q_max, q_max)
    return q.mul(scale), scale


def quantize_array_symmetric(
    x: np.ndarray,
    bits: int = 8,
) -> Tuple[np.ndarray, float]:
    """Numpy equivalent of :func:`quantize_tensor_symmetric`.

    Parameters
    ----------
    x : np.ndarray
        Array to quantise (any shape).
    bits : int
        Bit-width.

    Returns
    -------
    dequantised : np.ndarray
        Rounded array, same shape and dtype as *x*.
    scale : float
        Quantisation scale.
    """
    q_max = float(2 ** (bits - 1) - 1)
    abs_max = float(np.abs(x).max())
    if abs_max == 0.0:
        return x.copy(), 1.0

    scale = abs_max / q_max
    q = np.round(x / scale).clip(-q_max, q_max)
    return (q * scale).astype(x.dtype), scale


# ---------------------------------------------------------------------------
# Model quantisation
# ---------------------------------------------------------------------------

def quantize_model(
    model: SNNClassifier,
    bits: int = 8,
) -> SNNClassifier:
    """Return a deep copy of *model* with INT8-simulated ``Linear`` weights.

    Only ``weight`` parameters of ``torch.nn.Linear`` layers are quantised;
    biases and LIF-neuron parameters (``beta``, ``threshold``, …) are left
    at full precision.

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    bits : int
        Quantisation bit-width.

    Returns
    -------
    SNNClassifier
        Deep-copied model whose linear weights have been rounded to the
        nearest INT*bits* value and dequantised back to float.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()

    scale_log: list[str] = []
    with torch.no_grad():
        for name, module in model_q.named_modules():
            if isinstance(module, nn.Linear):
                dq, scale = quantize_tensor_symmetric(module.weight.data, bits)
                module.weight.data.copy_(dq)
                scale_log.append(f"{name}.weight scale={scale:.6f}")

    logger.info(
        "INT%d model: quantised %d Linear weight tensors — %s",
        bits, len(scale_log), "  ".join(scale_log),
    )
    return model_q


# ---------------------------------------------------------------------------
# CSP filter quantisation
# ---------------------------------------------------------------------------

def quantize_csp_filters(
    filters: Dict,
    bits: int = 8,
) -> Dict:
    """Return a copy of the CSP filter dict with INT8-simulated filter matrices.

    Parameters
    ----------
    filters : Dict
        Mapping ``(band_idx, pair) -> np.ndarray`` of shape
        ``(n_channels, 2 * m)``, as stored in :attr:`PairwiseCSP.filters_`.
    bits : int
        Quantisation bit-width.

    Returns
    -------
    Dict
        New dict with quantised (dequantised) filter matrices.
    """
    quantised: Dict = {}
    total_scale = 0.0

    for key, W in filters.items():
        Wq, scale = quantize_array_symmetric(W, bits)
        quantised[key] = Wq
        total_scale += scale

    logger.info(
        "INT%d CSP: quantised %d filter matrices  (mean scale %.6f)",
        bits, len(filters), total_scale / max(len(filters), 1),
    )
    return quantised


# ---------------------------------------------------------------------------
# Accuracy-loss summary
# ---------------------------------------------------------------------------

def quantization_report(
    fp32_acc: float,
    int8_acc: float,
    label: str = "test",
) -> None:
    """Log a one-line FP32 vs INT8 accuracy comparison.

    Parameters
    ----------
    fp32_acc : float
        Full-precision accuracy.
    int8_acc : float
        INT8-simulated accuracy.
    label : str
        Split name (``"val"`` or ``"test"``).
    """
    delta = (int8_acc - fp32_acc) * 100.0
    sign = "+" if delta >= 0 else ""
    logger.info(
        "Quantisation report (%s):  FP32 %.2f%%  INT8 %.2f%%  delta %s%.2f%%",
        label, fp32_acc * 100, int8_acc * 100, sign, delta,
    )
