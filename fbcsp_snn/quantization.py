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

Analog non-ideality injection (B15 Tier B)
-------------------------------------------
Digital bit-rounding above answers "how many bits does this need," but real
analog circuits don't fail from having fewer bits — they fail from thermal
noise, device-to-device mismatch, and comparator offset drift.  The
``inject_*_noise`` functions below add Gaussian noise sized as a fraction of
the nominal value, modelling three specific physical non-idealities:

- :func:`inject_weight_noise_tensor` / :func:`inject_weight_noise_array` —
  conductance variation in an analog crossbar array (CSP or SNN weights).
- :func:`inject_beta_noise` — RC time-constant mismatch across fabricated
  leaky-integrator circuits (per-neuron membrane decay).

All noise functions return a modified copy and accept a ``seed`` for
reproducibility; callers doing Monte Carlo repetition (B15 Tier C) should
vary the seed across repeats and aggregate mean/std, not rely on any single
noisy draw.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

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


def inject_csp_filter_noise(
    filters: Dict,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> Dict:
    """Return a copy of the CSP filter dict with Gaussian noise injected.

    Mirrors :func:`quantize_csp_filters`'s structure but adds noise (models
    analog crossbar conductance variation) instead of rounding to a bit-width.

    Parameters
    ----------
    filters : Dict
        Mapping ``(band_idx, pair) -> np.ndarray``, as stored in
        :attr:`PairwiseCSP.filters_`.
    sigma_frac : float
        Per-matrix noise std as a fraction of that matrix's peak magnitude.
    seed : Optional[int]
        Base RNG seed; incremented per matrix so different (band, pair)
        filters don't get identical noise patterns.

    Returns
    -------
    Dict
        New dict with noise-injected filter matrices.
    """
    noisy: Dict = {}
    for i, (key, W) in enumerate(filters.items()):
        layer_seed = None if seed is None else seed + i
        noisy[key] = inject_weight_noise_array(W, sigma_frac, seed=layer_seed)

    logger.info(
        "Weight-noise CSP: injected sigma_frac=%.4f into %d filter matrices",
        sigma_frac, len(filters),
    )
    return noisy


def inject_ea_whitener_noise(
    ea_whiteners: Dict,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> Dict:
    """Return a copy of the Euclidean Alignment whitener dict with noise injected.

    The EA whitener (``R^{-1/2}``, per band) is the same mathematical class
    as the CSP filters — a linear transform / analog crossbar matrix
    multiply — but was excluded from :func:`inject_csp_filter_noise`'s sweep
    since it's a separate stored artefact (`PairwiseCSP.ea_whiteners_`, not
    `PairwiseCSP.filters_`). This closes that gap.

    Parameters
    ----------
    ea_whiteners : Dict
        Mapping ``band_idx -> np.ndarray``, as stored in
        :attr:`PairwiseCSP.ea_whiteners_`.
    sigma_frac : float
        Per-matrix noise std as a fraction of that matrix's peak magnitude.
    seed : Optional[int]
        Base RNG seed; incremented per band.

    Returns
    -------
    Dict
        New dict with noise-injected whitener matrices.
    """
    noisy: Dict = {}
    for i, (key, R) in enumerate(ea_whiteners.items()):
        band_seed = None if seed is None else seed + i
        noisy[key] = inject_weight_noise_array(R, sigma_frac, seed=band_seed)

    logger.info(
        "Weight-noise EA whitener: injected sigma_frac=%.4f into %d matrices",
        sigma_frac, len(ea_whiteners),
    )
    return noisy


def inject_znorm_noise(
    mean: np.ndarray,
    std: np.ndarray,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to z-normalisation mean/std (analog reference drift).

    Models imprecision in the analog reference voltage (mean, typically a
    subtraction stage) and gain (1/std, typically a multiplicative stage) of
    a physical z-normalisation circuit.

    Parameters
    ----------
    mean : np.ndarray
        Per-feature mean, shape ``(n_features,)``, as stored in
        :attr:`fbcsp_snn.preprocessing.ZNormaliser.mean_`.
    std : np.ndarray
        Per-feature std, shape ``(n_features,)``, as stored in
        :attr:`fbcsp_snn.preprocessing.ZNormaliser.std_`.
    sigma_frac : float
        Noise std as a fraction of each array's peak magnitude.
    seed : Optional[int]
        Base RNG seed; ``std``'s noise uses ``seed + 1`` so mean/std don't
        get identical noise draws.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Noisy ``(mean, std)``. ``std`` is floored at ``1e-8`` (matching
        `ZNormaliser.fit`'s own epsilon) to avoid division by zero/negative
        scale downstream.
    """
    std_seed = None if seed is None else seed + 1
    mean_noisy = inject_weight_noise_array(mean, sigma_frac, seed=seed)
    std_noisy = inject_weight_noise_array(std, sigma_frac, seed=std_seed)
    std_noisy = np.maximum(std_noisy, 1e-8)

    logger.info(
        "Weight-noise Z-norm: injected sigma_frac=%.4f into mean/std (%d features)",
        sigma_frac, mean.shape[0],
    )
    return mean_noisy, std_noisy


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


# ---------------------------------------------------------------------------
# Analog non-ideality injection (B15 Tier B)
# ---------------------------------------------------------------------------

def inject_weight_noise_tensor(
    x: torch.Tensor,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise to a weight tensor (conductance variation).

    Parameters
    ----------
    x : torch.Tensor
        Weight tensor, any shape.
    sigma_frac : float
        Noise standard deviation as a fraction of ``max(|x|)`` (e.g. ``0.05``
        = noise std is 5% of the tensor's peak magnitude).
    seed : Optional[int]
        RNG seed for reproducibility across Monte Carlo repeats.

    Returns
    -------
    torch.Tensor
        Noisy copy of *x*, same shape and dtype.  *x* is not modified.
    """
    abs_max = x.abs().max().item()
    if abs_max == 0.0 or sigma_frac == 0.0:
        return x.clone()
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return x + noise * (sigma_frac * abs_max)


def inject_weight_noise_array(
    x: np.ndarray,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Numpy equivalent of :func:`inject_weight_noise_tensor`.

    Parameters
    ----------
    x : np.ndarray
        Weight array, any shape.
    sigma_frac : float
        Noise standard deviation as a fraction of ``max(|x|)``.
    seed : Optional[int]
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy copy of *x*, same shape and dtype.
    """
    abs_max = float(np.abs(x).max())
    if abs_max == 0.0 or sigma_frac == 0.0:
        return x.copy()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape).astype(x.dtype)
    return (x + noise * (sigma_frac * abs_max)).astype(x.dtype)


def inject_model_weight_noise(
    model: SNNClassifier,
    sigma_frac: float,
    seed: Optional[int] = None,
    include_bias: bool = True,
) -> SNNClassifier:
    """Return a deep copy of *model* with Gaussian noise added to Linear
    weights (and, by default, biases).

    Unlike :func:`quantize_model` (which intentionally excludes biases for a
    bit-precision reason — "INT8 biases add hardware complexity for minimal
    benefit"), biases are included here by default: that rationale doesn't
    carry over to analog noise injection, since biases are just as
    physically-realised (offset currents/voltages) as weights and subject to
    the same device mismatch. Set ``include_bias=False`` to restrict noise
    to weights only, matching the old scope.

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    sigma_frac : float
        Per-tensor noise std as a fraction of that tensor's peak magnitude.
    seed : Optional[int]
        Base RNG seed. Each weight/bias tensor gets its own offset so
        multi-layer models don't get identical noise patterns.
    include_bias : bool
        If ``True`` (default), also perturb ``Linear.bias``. If ``False``,
        weights only (the old scope, matching ``quantize_model``).

    Returns
    -------
    SNNClassifier
        Deep-copied, noise-injected model.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()

    draw = 0
    with torch.no_grad():
        for module in model_q.modules():
            if isinstance(module, nn.Linear):
                w_seed = None if seed is None else seed + draw
                draw += 1
                noisy_w = inject_weight_noise_tensor(
                    module.weight.data, sigma_frac, seed=w_seed
                )
                module.weight.data.copy_(noisy_w)

                if include_bias and module.bias is not None:
                    b_seed = None if seed is None else seed + draw
                    draw += 1
                    noisy_b = inject_weight_noise_tensor(
                        module.bias.data, sigma_frac, seed=b_seed
                    )
                    module.bias.data.copy_(noisy_b)

    return model_q


def inject_beta_noise(
    model: SNNClassifier,
    sigma_frac: float,
    seed: Optional[int] = None,
) -> SNNClassifier:
    """Perturb each LIF neuron's membrane decay (beta) with per-neuron noise.

    Models RC time-constant mismatch across fabricated analog leaky
    integrators — a source of real-world non-ideality that weight
    quantisation alone never touches (``quantize_model`` explicitly leaves
    LIF parameters at full precision).

    Parameters
    ----------
    model : SNNClassifier
        Source model.  Not modified.
    sigma_frac : float
        Noise std as a fraction of the nominal beta value (e.g. ``0.05`` =
        5% per-neuron variation).
    seed : Optional[int]
        RNG seed for reproducibility.

    Returns
    -------
    SNNClassifier
        Deep-copied model with per-neuron beta perturbed on both LIF layers,
        clamped to ``[0.01, 0.999]`` to stay in a physically valid decay range.

    Note
    ----
    Assumes snnTorch's ``Leaky.beta`` accepts a tensor shaped to the neuron
    count, which is documented snnTorch behaviour for heterogeneous/learnable
    beta. Not verified against an installed snnTorch on this machine (torch/
    snntorch aren't available locally) — confirm on the first Puhti run
    before relying on this for published results.
    """
    model_q = copy.deepcopy(model)
    model_q.eval()
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    with torch.no_grad():
        for lif, n_neurons in (
            (model_q.lif1, model_q.n_hidden),
            (model_q.lif2, model_q.n_output),
        ):
            nominal = lif.beta if torch.is_tensor(lif.beta) else torch.full(
                (n_neurons,), float(lif.beta)
            )
            noise = torch.randn(n_neurons, generator=gen) * sigma_frac * nominal
            lif.beta = (nominal + noise).clamp(0.01, 0.999)

    return model_q
