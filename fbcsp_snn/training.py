"""Training loop with AdamW, AMP, early stopping, and best-model checkpointing.

Design
------
- Mini-batch SGD over trials; trials are shuffled at every epoch.
- AMP (``torch.amp.autocast`` + ``GradScaler``) is enabled on CUDA and
  disabled on CPU (autocast over CPU uses bfloat16 but has no throughput
  benefit on non-Apple hardware).
- Early stopping monitors *validation accuracy* with a configurable patience
  and warmup period so the model has time to escape random-init noise before
  stopping kicks in.
- The best model state (by val accuracy) is kept in memory and optionally
  saved to ``fold_dir/best_model.pt``.  At the end of training the model is
  restored to that best state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast   # torch >= 2.2
except ImportError:
    from torch.cuda.amp import GradScaler        # torch 2.1.x
    from torch import autocast                   # type: ignore[assignment]

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.losses import make_target_spikes, van_rossum_loss
from fbcsp_snn.model import SNNClassifier

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Metrics produced by one training fold.

    Attributes
    ----------
    best_val_acc : float
        Best validation accuracy seen during training.
    best_epoch : int
        0-indexed epoch at which best val accuracy was achieved.
    stopped_epoch : int
        0-indexed epoch at which training stopped (may equal ``epochs - 1``
        if early stopping never triggered).
    train_loss_history : List[float]
        Mean training loss per epoch.
    val_acc_history : List[float]
        Validation accuracy per epoch.
    """

    best_val_acc: float = 0.0
    best_epoch: int = 0
    stopped_epoch: int = 0
    train_loss_history: List[float] = field(default_factory=list)
    val_acc_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor validation accuracy and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping.
    warmup : int
        Minimum number of epochs before early stopping can trigger.

    Attributes
    ----------
    best_acc : float
        Best validation accuracy seen so far.
    best_epoch : int
        Epoch (0-indexed) at which best accuracy occurred.
    counter : int
        Epochs since last improvement.
    """

    def __init__(self, patience: int = 100, warmup: int = 100) -> None:
        self.patience = patience
        self.warmup = warmup
        self.best_acc: float = -1.0
        self.best_epoch: int = 0
        self.counter: int = 0

    def update(self, val_acc: float, epoch: int) -> Tuple[bool, bool]:
        """Update state with the latest validation accuracy.

        Parameters
        ----------
        val_acc : float
            Validation accuracy for the current epoch.
        epoch : int
            Current epoch index (0-indexed).

        Returns
        -------
        improved : bool
            ``True`` if *val_acc* is a new best.
        should_stop : bool
            ``True`` if training should be terminated.
        """
        improved = val_acc > self.best_acc
        if improved:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        should_stop = (epoch >= self.warmup) and (self.counter >= self.patience)
        return improved, should_stop


# ---------------------------------------------------------------------------
# Evaluation helper (no gradient)
# ---------------------------------------------------------------------------

def evaluate_model(
    model: SNNClassifier,
    spikes: torch.Tensor,
    y_0idx: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, np.ndarray]:
    """Run inference and compute accuracy.

    Parameters
    ----------
    model : SNNClassifier
        Model to evaluate.  Must already be on *device*.
    spikes : torch.Tensor
        Spike tensor ``(T, n_trials, n_features)``.  Can be on any device;
        batches are moved to *device* inside the loop.
    y_0idx : np.ndarray
        True labels, shape ``(n_trials,)``.  **0-indexed.**
    device : torch.device
        Inference device.
    batch_size : int
        Number of trials per inference batch.

    Returns
    -------
    accuracy : float
        Fraction of correctly classified trials.
    predictions : np.ndarray
        Predicted class indices (0-indexed), shape ``(n_trials,)``.
    """
    model.eval()
    n_trials = spikes.shape[1]
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            batch = spikes[:, start : start + batch_size, :].to(device)
            spk_out, _ = model(batch)
            preds = model.decode(spk_out)
            all_preds.append(preds.cpu().numpy())

    predictions = np.concatenate(all_preds)
    accuracy = float((predictions == y_0idx).mean())
    return accuracy, predictions


# ---------------------------------------------------------------------------
# Full fold training
# ---------------------------------------------------------------------------

def train_fold(
    spikes_train: torch.Tensor,
    y_train: np.ndarray,
    spikes_val: torch.Tensor,
    y_val: np.ndarray,
    model: SNNClassifier,
    n_classes: int,
    population_per_class: int,
    *,
    spike_prob: float = 0.7,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    epochs: int = 1000,
    patience: int = 100,
    warmup: int = 100,
    tau_vr: float = 10.0,
    batch_size: int = 32,
    max_time_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    fold_dir: Optional[Path] = None,
    log_every: int = 10,
    lr_scheduler: str = "none",
    lr_min: float = 1e-5,
    lr_scheduler_patience: int = 30,
    lr_scheduler_factor: float = 0.5,
    activity_reg: float = 0.0,
    target_spike_rate: float = 0.1,
) -> FoldResult:
    """Train an SNN on one CV fold and return performance metrics.

    Parameters
    ----------
    spikes_train : torch.Tensor
        Training spike tensor ``(T, n_train, n_features)``.
    y_train : np.ndarray
        Training labels, shape ``(n_train,)``.  **0-indexed.**
    spikes_val : torch.Tensor
        Validation spike tensor ``(T, n_val, n_features)``.
    y_val : np.ndarray
        Validation labels, shape ``(n_val,)``.  **0-indexed.**
    model : SNNClassifier
        Freshly initialised model (will be moved to *device* if not already).
    n_classes : int
        Number of MI classes.
    population_per_class : int
        Output population size per class.
    spike_prob : float
        Target spiking probability for the correct class in Van Rossum targets.
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight-decay coefficient.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience (epochs without val-acc improvement).
    warmup : int
        Minimum epochs before early stopping activates.
    tau_vr : float
        Van Rossum loss time constant (timesteps).
    batch_size : int
        Mini-batch size (number of trials per gradient step).
    max_time_steps : Optional[int]
        If given, truncate spikes to the first *max_time_steps* timesteps.
        Useful during development to reduce per-epoch wall time.
    device : Optional[torch.device]
        Training device.  Defaults to :data:`fbcsp_snn.DEVICE`.
    fold_dir : Optional[Path]
        Directory to save ``best_model.pt``.  Created if absent.
    log_every : int
        Log metrics every this many epochs.
    activity_reg : float
        Coefficient for hidden-layer firing-rate regularisation.  ``0.0``
        disables it.  Adds ``activity_reg * (mean_hidden_rate - target)²``
        to the loss, penalising deviation from *target_spike_rate*.
    target_spike_rate : float
        Target mean spike rate for the hidden layer (spikes per neuron per
        timestep).  Typical range 0.05–0.2.
    lr_scheduler : str
        LR schedule type: ``"plateau"``, ``"cosine"``, or ``"none"``.
        ``"plateau"`` reduces LR by *lr_scheduler_factor* after
        *lr_scheduler_patience* epochs without val-accuracy improvement.
        ``"cosine"`` anneals from *lr* to *lr_min* over *epochs* steps.
    lr_min : float
        Minimum LR floor (used by both schedulers).
    lr_scheduler_patience : int
        Plateau scheduler: epochs without improvement before reducing LR.
    lr_scheduler_factor : float
        Plateau scheduler: multiplicative LR reduction factor.

    Returns
    -------
    FoldResult
        Metrics for the fold.  The model is restored to its best-val-accuracy
        state before returning.
    """
    if device is None:
        device = DEVICE

    # Optional time truncation (development / test speedup only)
    if max_time_steps is not None:
        spikes_train = spikes_train[:max_time_steps]
        spikes_val   = spikes_val[:max_time_steps]

    T = spikes_train.shape[0]
    n_train = spikes_train.shape[1]

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_scheduler_factor,
            patience=lr_scheduler_patience, min_lr=lr_min,
        )
        logger.info(
            "LR scheduler: ReduceLROnPlateau  factor=%.2f  patience=%d  min_lr=%.2e",
            lr_scheduler_factor, lr_scheduler_patience, lr_min,
        )
    elif lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min,
        )
        logger.info(
            "LR scheduler: CosineAnnealingLR  T_max=%d  eta_min=%.2e", epochs, lr_min,
        )
    else:
        scheduler = None

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    es = EarlyStopping(patience=patience, warmup=warmup)
    best_state: dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    y_tensor = torch.from_numpy(y_train)   # stays on CPU; moved per batch

    train_loss_history: list[float] = []
    val_acc_history: list[float] = []
    stopped_epoch = epochs - 1

    for epoch in range(epochs):
        # ---- training ----
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            batch_spikes = spikes_train[:, idx, :].to(device)   # (T, B, F)
            batch_y      = y_tensor[idx].long().to(device)       # (B,)

            target = make_target_spikes(
                batch_y, n_classes, population_per_class, T, spike_prob
            ).to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                spk_out, _, spk_hidden = model(batch_spikes, return_hidden=True)
                loss = van_rossum_loss(spk_out, target, tau=tau_vr)
                if activity_reg > 0.0:
                    mean_rate = spk_hidden.float().mean()
                    loss = loss + activity_reg * (mean_rate - target_spike_rate) ** 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        train_loss_history.append(mean_loss)

        # ---- validation ----
        val_acc, _ = evaluate_model(model, spikes_val, y_val, device, batch_size)
        val_acc_history.append(val_acc)

        improved, should_stop = es.update(val_acc, epoch)

        if improved:
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            if fold_dir is not None:
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, fold_dir / "best_model.pt")

        # LR scheduler step — only after warmup so early exploration isn't cut short
        prev_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None and epoch >= warmup:
            if lr_scheduler == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_changed = current_lr != prev_lr

        if (epoch + 1) % log_every == 0 or should_stop or lr_changed:
            logger.info(
                "  epoch %3d/%d  loss %.5f  val_acc %.3f  best %.3f  lr %.2e%s",
                epoch + 1, epochs, mean_loss, val_acc, es.best_acc, current_lr,
                "  [LR reduced]" if lr_changed else "",
            )

        if should_stop:
            logger.info(
                "  Early stop at epoch %d  (best epoch %d  val_acc %.3f)",
                epoch + 1, es.best_epoch + 1, es.best_acc,
            )
            stopped_epoch = epoch
            break

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return FoldResult(
        best_val_acc=es.best_acc,
        best_epoch=es.best_epoch,
        stopped_epoch=stopped_epoch,
        train_loss_history=train_loss_history,
        val_acc_history=val_acc_history,
    )
