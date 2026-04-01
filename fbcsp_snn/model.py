"""2-layer LIF Spiking Neural Network with population-coded output.

Architecture
------------
::

    Input (T, batch, n_input)
      for t in 0 .. T-1:
        cur1 = fc1(x[t])                       # Linear(n_input, n_hidden)
        cur1 = drop1(cur1)                     # Dropout
        spk1, mem1 = lif1(cur1, mem1)          # LIF
        cur2 = fc2(spk1)                       # Linear(n_hidden, n_output)
        cur2 = drop2(cur2)                     # Dropout
        spk2, mem2 = lif2(cur2, mem2)          # LIF
      stack spk2, mem2 → (T, batch, n_output)

Output neurons are arranged in *n_classes* populations of *population_per_class*
neurons each.  During inference, spikes are summed over time and over each
population; the class with the highest total vote wins (winner-take-all).

``torch.compile`` is guarded behind a Triton availability check: Triton is
Linux-only and CUDA Graphs conflict with snnTorch's ``init_leaky()``, so when
compiling the model use ``mode="default"`` only.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

from fbcsp_snn import DEVICE, setup_logger

logger: logging.Logger = setup_logger(__name__)


class SNNClassifier(nn.Module):
    """Two-layer LIF SNN with population-coded output.

    Parameters
    ----------
    n_input : int
        Number of input features (= total CSP features after optional MIBIF
        selection).
    n_hidden : int
        Hidden layer width.
    n_classes : int
        Number of motor imagery classes.
    population_per_class : int
        Output neurons allocated per class.
    beta : float
        LIF membrane potential decay factor (shared across both layers).
    dropout_prob : float
        Dropout probability applied after each linear layer.
    use_bn : bool
        If ``True`` (default), insert ``BatchNorm1d`` after each ``Linear``
        layer.  BN is folded into the preceding Linear at deployment time via
        :func:`~fbcsp_snn.quantization.fold_batchnorm`, so it adds no extra
        operations on neuromorphic hardware.

    Attributes
    ----------
    n_output : int
        Total output neurons (= ``n_classes * population_per_class``).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 64,
        n_classes: int = 4,
        population_per_class: int = 20,
        beta: float = 0.95,
        dropout_prob: float = 0.5,
        use_bn: bool = True,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.population_per_class = population_per_class
        self.n_output = n_classes * population_per_class
        self.use_bn = use_bn
        self.surrogate_slope = surrogate_slope

        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        # Layer 1: Linear → BN → Dropout → LIF
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout_prob)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Layer 2: Linear → BN → Dropout → LIF
        self.fc2 = nn.Linear(n_hidden, self.n_output)
        self.bn2 = nn.BatchNorm1d(self.n_output) if use_bn else nn.Identity()
        self.drop2 = nn.Dropout(p=dropout_prob)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SNNClassifier — input: %d  hidden: %d  classes: %d  "
            "pop/class: %d  output: %d  use_bn: %s  surrogate_slope: %.1f  trainable params: %d",
            n_input, n_hidden, n_classes, population_per_class,
            self.n_output, use_bn, surrogate_slope, param_count,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run SNN simulation over all timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Input spike tensor, shape ``(T, batch, n_input)``.
        return_hidden : bool
            If ``True``, also return hidden layer spike trains as a third
            element (used for activity regularisation during training).

        Returns
        -------
        spk_out : torch.Tensor
            Output spike trains, shape ``(T, batch, n_output)``.
        mem_out : torch.Tensor
            Output membrane potential traces, shape ``(T, batch, n_output)``.
        spk_hidden : torch.Tensor  *(only when return_hidden=True)*
            Hidden layer spike trains, shape ``(T, batch, n_hidden)``.
        """
        T = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out_list: list[torch.Tensor] = []
        mem_out_list: list[torch.Tensor] = []
        spk_hidden_list: list[torch.Tensor] = []

        for t in range(T):
            # Layer 1: Linear → BN → Dropout → LIF
            cur1 = self.drop1(self.bn1(self.fc1(x[t])))
            spk1, mem1 = self.lif1(cur1, mem1)
            # Layer 2: Linear → BN → Dropout → LIF
            cur2 = self.drop2(self.bn2(self.fc2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out_list.append(spk2)
            mem_out_list.append(mem2)
            if return_hidden:
                spk_hidden_list.append(spk1)

        spk_out = torch.stack(spk_out_list, dim=0)   # (T, batch, n_output)
        mem_out = torch.stack(mem_out_list, dim=0)   # (T, batch, n_output)
        if return_hidden:
            return spk_out, mem_out, torch.stack(spk_hidden_list, dim=0)
        return spk_out, mem_out

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, spk_out: torch.Tensor) -> torch.Tensor:
        """Winner-take-all decoding from output spike trains.

        1. Sum spikes over time → ``(batch, n_output)``
        2. Reshape to ``(batch, n_classes, population_per_class)``
        3. Sum over population → ``(batch, n_classes)``
        4. Argmax → ``(batch,)`` predicted class (0-indexed)

        Parameters
        ----------
        spk_out : torch.Tensor
            Shape ``(T, batch, n_output)``.

        Returns
        -------
        torch.Tensor
            Predicted class indices, shape ``(batch,)``, dtype ``torch.long``.
        """
        spike_sums = spk_out.sum(dim=0)                              # (batch, n_output)
        spike_sums = spike_sums.view(-1, self.n_classes,
                                     self.population_per_class)     # (batch, C, pop)
        class_votes = spike_sums.sum(dim=-1)                        # (batch, C)
        return class_votes.argmax(dim=-1)                           # (batch,)


# ---------------------------------------------------------------------------
# Optional torch.compile guard (Linux / Triton only)
# ---------------------------------------------------------------------------

def maybe_compile(model: nn.Module) -> nn.Module:
    """Wrap *model* in ``torch.compile`` only when Triton is available.

    Triton is Linux-only; CUDA Graphs conflict with snnTorch's
    ``init_leaky()``, so ``mode="default"`` is used (not ``"reduce-overhead"``).

    Parameters
    ----------
    model : nn.Module
        Model to (optionally) compile.

    Returns
    -------
    nn.Module
        Compiled model or the original if Triton is unavailable.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        logger.info("torch.compile skipped (Triton not available)")
        return model

    # Triton PTX codegen is unreliable on Volta (V100, sm_70) with
    # PyTorch <= 2.1. Only compile on Ampere+ (sm_80+).
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.info(
                "torch.compile skipped (GPU is sm_%d0, Ampere sm_80+ required "
                "for stable Triton PTX codegen)",
                major,
            )
            return model

    compiled = torch.compile(model, mode="default")
    logger.info("torch.compile applied (Triton available, sm_80+)")
    return compiled
