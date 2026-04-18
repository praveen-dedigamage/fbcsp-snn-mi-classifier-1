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

class _CUBABlock(nn.Module):
    """Single CUBA-LIF layer: nn.Linear (with bias) + SLAYER CUBA neuron.

    Why nn.Linear instead of slayer.block.cuba.Dense:
    snnTorch Linear layers are trained WITH bias (PyTorch default).  SLAYER's
    Dense synapse has no bias parameter, so a naive weight-only transfer drops
    the bias term.  With T=1001 timesteps the bias accumulates as steady-state
    membrane potential V_ss = bias / (1-β) = bias / 0.05 = 20 × bias.
    Omitting it causes ~45 pp accuracy collapse.

    On physical Loihi 2, per-neuron bias maps to the neuron's bias-current
    register (a supported hardware feature), so this remains fully
    Loihi 2 compatible.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        threshold: float,
        current_decay: float,
        voltage_decay: float,
    ) -> None:
        _check_lava()
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=True)
        # slayer.neuron.cuba.Neuron takes individual keyword arguments
        self.neuron = slayer.neuron.cuba.Neuron(
            threshold     = threshold,
            current_decay = current_decay,
            voltage_decay = voltage_decay,
            requires_grad = False,
            graded_spike  = False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_in, T)

        Returns
        -------
        spk : (batch, n_out, T)
        """
        # Apply Linear across all timesteps:
        # einsum bIT, OI -> bOT  (I=in_features, O=out_features, T=time)
        cur = torch.einsum("bit,oi->bot", x, self.linear.weight)
        cur = cur + self.linear.bias.unsqueeze(0).unsqueeze(-1)  # broadcast bias
        return self.neuron(cur)


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
        lif_kwargs = dict(
            threshold     = 1.0,
            current_decay = 1.0,
            voltage_decay = float(1.0 - beta),
        )

        self.layer1 = _CUBABlock(n_input,       n_hidden,       **lif_kwargs)
        self.layer2 = _CUBABlock(n_hidden, self.n_output, **lif_kwargs)

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

        # Transfer weights AND biases from snnTorch Linear layers.
        # Bias MUST be included: snnTorch trains with bias=True (PyTorch default).
        # Steady-state bias contribution: V_ss = bias/(1-β) = 20×bias at β=0.95.
        # On Loihi 2, per-neuron bias maps to the neuron's hardware bias register.
        w1 = state_dict["fc1.weight"].cpu().float()
        b1 = state_dict["fc1.bias"].cpu().float()
        w2 = state_dict["fc2.weight"].cpu().float()
        b2 = state_dict["fc2.bias"].cpu().float()

        net.layer1.linear.weight.data.copy_(w1)
        net.layer1.linear.bias.data.copy_(b1)
        net.layer2.linear.weight.data.copy_(w2)
        net.layer2.linear.bias.data.copy_(b2)

        logger.info(
            "Weights+biases transferred — "
            "fc1 W%s b%s | fc2 W%s b%s",
            tuple(w1.shape), tuple(b1.shape),
            tuple(w2.shape), tuple(b2.shape),
        )

        net.eval()
        return net

    # ------------------------------------------------------------------
    # NETX export
    # ------------------------------------------------------------------

    def export_hdf5(self, path: str | Path) -> None:
        """Export architecture + weights to HDF5 (Loihi 2 / NETX-compatible layout).

        lava-dl 0.6.0 does not expose a public ``slayer.utils.io.save`` API.
        We write the HDF5 directly in the NETX schema so the file is loadable
        by ``lava.lib.dl.netx.hdf5.Network`` and suitable for an INRC
        application or manual Loihi 2 deployment.

        Parameters
        ----------
        path : str or Path
        """
        import h5py, numpy as np
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lif_params = {
            "type":           "CUBA",
            "threshold":      1.0,
            "current_decay":  1.0,
            "voltage_decay":  float(self.layer1.neuron.voltage_decay),
        }

        with h5py.File(path, "w") as f:
            f.attrs["n_layers"]  = 2
            f.attrs["n_input"]   = self.n_input
            f.attrs["n_hidden"]  = self.n_hidden
            f.attrs["n_output"]  = self.n_output
            f.attrs["n_classes"] = self.n_classes
            f.attrs["pop_per_class"] = self.population_per_class

            for i, (layer, name) in enumerate(
                [(self.layer1, "layer1"), (self.layer2, "layer2")]
            ):
                grp = f.create_group(name)
                grp["weight"] = layer.linear.weight.data.cpu().numpy()
                grp["bias"]   = layer.linear.bias.data.cpu().numpy()
                for k, v in lif_params.items():
                    grp.attrs[k] = v

        logger.info("NETX-compatible HDF5 saved → %s", path)

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
