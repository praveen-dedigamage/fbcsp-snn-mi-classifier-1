"""2-layer LIF Spiking Neural Network with population-coded output.

Architecture — standard (recurrent_hidden=False)
-------------------------------------------------
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

Architecture — recurrent hidden (recurrent_hidden=True)
--------------------------------------------------------
::

    Input (T, batch, n_input)
      spk1 = zeros(batch, n_hidden)            # recurrent spike state
      for t in 0 .. T-1:
        cur1 = fc1(x[t])                       # Linear(n_input, n_hidden)
        cur1 = drop1(cur1)                     # Dropout
        spk1, mem1 = rlif1(cur1, spk1, mem1)  # RLeaky: mem += W_rec @ spk1_prev
        cur2 = fc2(spk1)                       # Linear(n_hidden, n_output)
        cur2 = drop2(cur2)                     # Dropout
        spk2, mem2 = lif2(cur2, mem2)          # LIF (output unchanged)
      stack spk2, mem2 → (T, batch, n_output)

The recurrent weight matrix W_rec is (n_hidden × n_hidden), learnable.
Only the hidden layer is made recurrent; the output population layer
remains a standard LIF (recurrent feedback on a read-out is not useful).

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
    recurrent_hidden : bool
        If ``True``, replace the hidden LIF with a recurrent LIF
        (``snn.RLeaky``) that adds lateral spike feedback via a learnable
        ``(n_hidden × n_hidden)`` weight matrix.  The output layer remains
        a standard ``snn.Leaky``.  Default ``False``.

    Attributes
    ----------
    n_output : int
        Total output neurons (= ``n_classes * population_per_class``).
    recurrent_hidden : bool
        Whether the hidden layer uses recurrent connections.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 64,
        n_classes: int = 4,
        population_per_class: int = 20,
        beta: float = 0.95,
        dropout_prob: float = 0.5,
        recurrent_hidden: bool = False,
    ) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.population_per_class = population_per_class
        self.n_output = n_classes * population_per_class
        self.recurrent_hidden = recurrent_hidden

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layer 1: Linear → Dropout → LIF (or RLeaky)
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.drop1 = nn.Dropout(p=dropout_prob)
        if recurrent_hidden:
            # RLeaky: mem[t] = β·mem[t-1] + W_in·x[t] + W_rec·spk[t-1]
            # all_to_all=True + linear_features creates W_rec as nn.Linear(n_hidden, n_hidden, bias=False)
            self.lif1 = snn.RLeaky(
                beta=beta,
                spike_grad=spike_grad,
                all_to_all=True,
                linear_features=n_hidden,
                learn_recurrent=True,
            )
        else:
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Layer 2: Linear → Dropout → LIF  (always standard — output read-out)
        self.fc2 = nn.Linear(n_hidden, self.n_output)
        self.drop2 = nn.Dropout(p=dropout_prob)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SNNClassifier — input: %d  hidden: %d  classes: %d  "
            "pop/class: %d  output: %d  recurrent_hidden: %s  trainable params: %d",
            n_input, n_hidden, n_classes, population_per_class,
            self.n_output, recurrent_hidden, param_count,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run SNN simulation over all timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Input spike tensor, shape ``(T, batch, n_input)``.

        Returns
        -------
        spk_out : torch.Tensor
            Output spike trains, shape ``(T, batch, n_output)``.
        mem_out : torch.Tensor
            Output membrane potential traces, shape ``(T, batch, n_output)``.
        """
        T, batch, _ = x.shape
        mem2 = self.lif2.init_leaky()

        if self.recurrent_hidden:
            # RLeaky requires explicit (spk, mem) state initialisation
            spk1 = torch.zeros(batch, self.n_hidden, device=x.device)
            mem1 = torch.zeros(batch, self.n_hidden, device=x.device)
        else:
            mem1 = self.lif1.init_leaky()

        spk_out_list: list[torch.Tensor] = []
        mem_out_list: list[torch.Tensor] = []

        for t in range(T):
            # Layer 1
            cur1 = self.drop1(self.fc1(x[t]))
            if self.recurrent_hidden:
                # spk1 from previous step is fed back through W_rec
                spk1, mem1 = self.lif1(cur1, spk1, mem1)
            else:
                spk1, mem1 = self.lif1(cur1, mem1)
            # Layer 2 (standard LIF read-out)
            cur2 = self.drop2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out_list.append(spk2)
            mem_out_list.append(mem2)

        spk_out = torch.stack(spk_out_list, dim=0)   # (T, batch, n_output)
        mem_out = torch.stack(mem_out_list, dim=0)   # (T, batch, n_output)
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
