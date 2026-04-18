"""Lava-DL (SLAYER) port of SNNClassifier for Loihi 2 simulation.

This module re-implements the two-layer LIF network from model.py using
lava.lib.dl.slayer so it can be:

  1. Run through Lava's PyTorch-based SLAYER simulator (accuracy verification).
  2. Exported to the .net HDF5 format consumed by lava.lib.dl.netx (Loihi 2
     deployment / bit-accurate simulation).

Neuron-model mapping
--------------------
snnTorch ``Leaky(beta=β)`` dynamics (no synaptic-current term):

    V[t] = β · V[t-1] + W · x[t]

SLAYER CUBA dynamics:

    I[t] = (1 - α_curr) · I[t-1] + W · x[t]
    V[t] = (1 - α_volt) · V[t-1] + I[t]

Setting α_curr = 1 (instant current decay) collapses the two-equation CUBA
model to a single-equation LIF identical to snnTorch Leaky:

    I[t] = W · x[t]                     # no synaptic integration
    V[t] = (1 - α_volt) · V[t-1] + I[t] = β · V[t-1] + W · x[t]   ✓

Weight transfer
---------------
snnTorch  fc1.weight : (n_hidden, n_input)
SLAYER    layer1.synapse.weight : (n_hidden, n_input, 1, 1, 1)

Biases from the Linear layers are intentionally dropped — Loihi 2 does not
support per-neuron DC-current injection in the standard dense-layer mapping.

Dimension convention
--------------------
snnTorch : (T, batch, features)  →  time-first
SLAYER   : (batch, features, T)  →  channel-last / time-last

All conversions are handled internally.

Note
----
This module is imported in the *.venv_lava* environment (torch 2.3.1 + lava-dl
0.6.0).  It does NOT import snnTorch or moabb.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Try importing lava — fail gracefully so the rest of fbcsp_snn still loads
# ---------------------------------------------------------------------------
try:
    import lava.lib.dl.slayer as slayer
    _LAVA_AVAILABLE = True
except ImportError:
    slayer = None           # type: ignore[assignment]
    _LAVA_AVAILABLE = False
    logger.warning("lava.lib.dl.slayer not available — LavaNetwork cannot be used. "
                   "Activate .venv_lava to use this module.")


def _check_lava() -> None:
    if not _LAVA_AVAILABLE:
        raise ImportError(
            "lava-dl is not installed in this environment.  "
            "Activate .venv_lava:  "
            "source /scratch/project_2003397/praveen/.venv_lava/bin/activate"
        )


# ---------------------------------------------------------------------------
# SLAYER network
# ---------------------------------------------------------------------------

class LavaNetwork(nn.Module):
    """Two-layer CUBA-LIF network equivalent to SNNClassifier.

    Parameters
    ----------
    n_input : int
    n_hidden : int
    n_classes : int
    population_per_class : int
    beta : float
        LIF membrane decay (same as snnTorch Leaky beta).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_classes: int,
        population_per_class: int,
        beta: float = 0.95,
    ) -> None:
        _check_lava()
        super().__init__()

        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.population_per_class = population_per_class
        self.n_output = n_classes * population_per_class

        # alpha_volt = 1 - beta  (voltage decay rate)
        # alpha_curr = 1.0       (instant synaptic current → matches snnTorch Leaky)
        neuron_params = {
            'threshold':      1.0,
            'current_decay':  1.0,
            'voltage_decay':  float(1.0 - beta),
            'requires_grad':  False,
            'graded_spike':   False,
        }

        # Layer 1: Dense synapse + CUBA LIF
        self.layer1 = slayer.block.cuba.Dense(
            neuron_params, n_input, n_hidden,
            weight_norm=False, delay=False,
        )

        # Layer 2: Dense synapse + CUBA LIF
        self.layer2 = slayer.block.cuba.Dense(
            neuron_params, n_hidden, self.n_output,
            weight_norm=False, delay=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "LavaNetwork — input:%d  hidden:%d  output:%d  "
            "voltage_decay:%.4f  params:%d",
            n_input, n_hidden, self.n_output, 1.0 - beta, n_params,
        )

    # ------------------------------------------------------------------
    # Forward  (SLAYER convention: batch × features × time)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the SLAYER simulation.

        Parameters
        ----------
        x : torch.Tensor
            Input spikes, shape ``(batch, n_input, T)``.

        Returns
        -------
        spk1 : torch.Tensor  shape (batch, n_hidden, T)
        spk2 : torch.Tensor  shape (batch, n_output, T)
        """
        spk1 = self.layer1(x)    # (batch, n_hidden, T)
        spk2 = self.layer2(spk1) # (batch, n_output, T)
        return spk1, spk2

    # ------------------------------------------------------------------
    # Decoder  (WTA identical to pipeline.py)
    # ------------------------------------------------------------------

    def decode(self, spk2: torch.Tensor) -> torch.Tensor:
        """Winner-take-all decoding over population-coded output.

        Parameters
        ----------
        spk2 : torch.Tensor
            Output spikes ``(batch, n_output, T)``.

        Returns
        -------
        preds : torch.Tensor
            Predicted class indices ``(batch,)``.
        """
        votes = spk2.sum(dim=-1)                              # (batch, n_output)
        votes = votes.view(-1, self.n_classes,
                           self.population_per_class).sum(-1) # (batch, n_classes)
        return votes.argmax(dim=1)

    # ------------------------------------------------------------------
    # Weight transfer from snnTorch checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_snntorch(
        cls,
        state_dict: Dict[str, torch.Tensor],
        params: Dict[str, Any],
    ) -> "LavaNetwork":
        """Build a LavaNetwork and copy weights from a snnTorch state dict.

        Parameters
        ----------
        state_dict : dict
            ``torch.load("best_model.pt")`` result.
        params : dict
            Fold pipeline_params.json dict (for n_input, n_hidden, etc.).

        Returns
        -------
        LavaNetwork (eval mode, on CPU)
        """
        n_input  = params["n_input_features"]
        n_hidden = params.get("hidden_neurons", 64)
        n_classes = params.get("n_classes", 4)
        pop      = params.get("population_per_class", 20)
        beta     = params.get("beta", 0.95)

        net = cls(n_input, n_hidden, n_classes, pop, beta)

        # snnTorch fc1.weight: (n_hidden, n_input)
        # SLAYER synapse.weight: (out, in, 1, 1, 1)
        w1 = state_dict["fc1.weight"].cpu().float()
        w2 = state_dict["fc2.weight"].cpu().float()

        net.layer1.synapse.weight.data.copy_(
            w1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        net.layer2.synapse.weight.data.copy_(
            w2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        logger.info(
            "Weights transferred — fc1 %s → layer1 %s | fc2 %s → layer2 %s",
            tuple(w1.shape),
            tuple(net.layer1.synapse.weight.shape),
            tuple(w2.shape),
            tuple(net.layer2.synapse.weight.shape),
        )

        net.eval()
        return net

    # ------------------------------------------------------------------
    # NETX export
    # ------------------------------------------------------------------

    def export_hdf5(self, path: str | Path) -> None:
        """Export the network to the Lava NETX .net HDF5 format.

        The resulting file can be loaded with ``lava.lib.dl.netx.hdf5.Network``
        for bit-accurate Loihi 2 simulation or on-chip deployment.

        Parameters
        ----------
        path : str or Path
            Output path, e.g. ``"Results_lava/network_S1_fold0.net"``
        """
        _check_lava()
        from lava.lib.dl.slayer.utils.io import save as slayer_save
        slayer_save(self, str(path))
        logger.info("NETX .net file saved → %s", path)

    # ------------------------------------------------------------------
    # Resource summary
    # ------------------------------------------------------------------

    def resource_summary(self) -> Dict[str, int]:
        """Return Loihi 2 resource counts for the paper.

        Returns
        -------
        dict with keys: neurons, synapses, max_fan_in
        """
        n_syn = self.n_input * self.n_hidden + self.n_hidden * self.n_output
        return {
            "neurons":     self.n_hidden + self.n_output,
            "synapses":    n_syn,
            "max_fan_in":  max(self.n_input, self.n_hidden),
        }
