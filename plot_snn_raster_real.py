"""Plot a REAL SNN spike-flow raster (input -> hidden -> output layers) for
one real BNCI2014-001 trial, using an ACTUAL trained fold produced by the
CURRENT documented pipeline (fixed six-band, csp_components_per_band=8,
adaptive MIBIF) -- e.g. from run_puhti_snn_raster_demo.sh, pulled back from
Puhti into <results-dir>/Subject_<id>/fold_<n>/.

Three panels, top to bottom, sharing one time axis:
  1. Input spikes  -- the adaptive-threshold encoder's output AFTER MIBIF
     selection (i.e. exactly what feeds fc1). Real filter bank -> real
     dual-end CSP (fitted, training-fold only) -> real z-norm (fitted,
     training-fold only) -> real encode_tensor -> real MIBIFSelector.transform.
  2. Hidden layer spikes -- SNNClassifier's first LIF layer, 64 neurons
     (config.py default), obtained via forward(..., return_hidden=True).
  3. Output layer spikes -- SNNClassifier's population-coded output layer
     (n_classes * population_per_class neurons), the same spk_out that
     WTA-decoding sums to produce the predicted class.

Each panel uses the same first-3/separator/last-3 raster design as the
other stage figures (plot_encoding_signal_real.py etc.), scaled to that
panel's own neuron count.

No data leakage: CSP/z-norm/MIBIF/SNN were all fit on the training session
only (by the training run itself); this script only transforms one
held-out test trial through those already-fitted parameters.

Requires the full stack (torch, snntorch, mne, moabb) -- run with the
project's real environment (e.g. .venv_realdata), not the lightweight
synthetic scripts' environment.

Usage:
    python plot_snn_raster_real.py \\
        --results-dir Results_snn_raster_demo --subject-id 1 --fold 0 \\
        --trial-idx 0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fbcsp_snn.datasets import load_moabb
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier

# mne/moabb apply their own matplotlib style on import -- reset later,
# right before plotting, to match the other pipeline-stage figures exactly.

CLASS_NAMES = {1: "Left Hand", 2: "Right Hand", 3: "Feet", 4: "Tongue"}


def _concat_projections(proj: dict) -> np.ndarray:
    """Matches fbcsp_snn/pipeline.py's _concat_projections exactly."""
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def _raster_panel(ax, spikes_1d_by_feat: np.ndarray, time_s: np.ndarray,
                   label_prefix: str, title: str) -> None:
    """Draw one first-3/separator/last-3 raster panel.

    Parameters
    ----------
    spikes_1d_by_feat : np.ndarray
        Shape (n_features, n_timesteps), binary.
    """
    n = spikes_1d_by_feat.shape[0]
    if n >= 7:
        rows = [("signal", 0, f"{label_prefix} 0"),
                ("signal", 1, f"{label_prefix} 1"),
                ("signal", 2, f"{label_prefix} 2"),
                ("separator", None, "\n".join(["·"] * 5)),
                ("signal", n - 3, f"{label_prefix} N-3"),
                ("signal", n - 2, f"{label_prefix} N-2"),
                ("signal", n - 1, f"{label_prefix} N-1")]
    else:
        rows = [("signal", i, f"{label_prefix} {i}") for i in range(n)]

    n_rows = len(rows)
    offset_step = 1.0
    tick_height = 0.7
    offsets = [(n_rows - 1 - row_i) * offset_step for row_i in range(n_rows)]

    yticks, yticklabels = [], []
    separator_offset = None
    signal_row_count = 0
    for row_i, (kind, idx, label) in enumerate(rows):
        offset = offsets[row_i]
        if kind == "separator":
            separator_offset = offset
            ax.hlines(offset, time_s[0], time_s[-1], linestyle=":",
                      linewidth=2.5, color="0.5")
        else:
            spike_times = time_s[spikes_1d_by_feat[idx] > 0]
            color = f"C{signal_row_count}"
            ax.vlines(spike_times, offset, offset + tick_height,
                      linewidth=0.8, color=color)
            signal_row_count += 1
        yticks.append(offset)
        yticklabels.append(label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.tick_params(axis="y", length=6, width=1.0, direction="out")

    if separator_offset is not None:
        for tick in ax.yaxis.get_major_ticks():
            if np.isclose(tick.get_loc(), separator_offset):
                tick.tick1line.set_visible(False)
                break

    ax.set_xticks([])
    ax.set_title(title, fontsize=13)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="Results_snn_raster_demo")
    ap.add_argument("--subject-id", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--trial-idx", type=int, default=0,
                     help="Index into the TEST session's trials to plot.")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--output", default="figures/snn_raster_real.png")
    args = ap.parse_args()

    fold_dir = os.path.join(args.results_dir, f"Subject_{args.subject_id}",
                             f"fold_{args.fold}")
    print(f"Loading real trained fold artifacts from {fold_dir} ...")
    with open(os.path.join(fold_dir, "pipeline_params.json")) as f:
        params = json.load(f)
    with open(os.path.join(fold_dir, "csp_filters.pkl"), "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(os.path.join(fold_dir, "znorm.pkl"), "rb") as f:
        znorm: ZNormaliser = pickle.load(f)
    mibif: MIBIFSelector | None = None
    mibif_path = os.path.join(fold_dir, "mibif.pkl")
    if os.path.exists(mibif_path):
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    bands = [tuple(b) for b in params["bands"]]
    encoder_type = params.get("encoder_type", "delta")
    base_thresh = params.get("base_thresh", 0.001)
    adapt_inc = params.get("adapt_inc", 0.6)
    decay = params.get("decay", 0.95)
    n_classes = params["n_classes"]
    hidden_neurons = params.get("hidden_neurons", 64)
    population_per_class = params.get("population_per_class", 20)
    beta = params.get("beta", 0.95)
    print(f"  bands={bands}  encoder={encoder_type}  "
          f"hidden={hidden_neurons}  pop/class={population_per_class}")

    print(f"Loading real MOABB data: BNCI2014_001, subject {args.subject_id} ...")
    _, _, X_test, y_test = load_moabb(params["dataset"], args.subject_id)
    trial_idx = args.trial_idx
    y_test_one = y_test[trial_idx:trial_idx + 1]
    X_test_one = X_test[trial_idx:trial_idx + 1]
    del X_test, y_test
    gc.collect()

    print("Applying real filter bank + fitted CSP + fitted z-norm "
          "(training-fold-fitted parameters, held-out test trial) ...")
    Xb_test_one = apply_filter_bank(X_test_one, bands, args.sfreq)
    proj_test_one = csp.transform(Xb_test_one)
    X_concat_test_one = _concat_projections(proj_test_one).astype(np.float32)
    X_norm_test = znorm.transform(X_concat_test_one)  # (1, n_feat, n_samples)
    del Xb_test_one, proj_test_one, X_concat_test_one
    gc.collect()

    print(f"Encoding spikes (base_thresh={base_thresh}, adapt_inc={adapt_inc}, "
          f"decay={decay}) ...")
    X_t = torch.from_numpy(X_norm_test).permute(2, 0, 1)  # (T, 1, n_feat)
    spikes_pre_mibif = encode_tensor(X_t, base_thresh, adapt_inc, decay,
                                      encoder_type)

    if mibif is not None:
        spikes_in = mibif.transform(spikes_pre_mibif)
        print(f"  MIBIF selected {spikes_in.shape[2]} / "
              f"{spikes_pre_mibif.shape[2]} features")
    else:
        spikes_in = spikes_pre_mibif
    n_input = spikes_in.shape[2]

    print(f"Loading trained SNNClassifier (n_input={n_input}, "
          f"n_hidden={hidden_neurons}, n_classes={n_classes}, "
          f"population_per_class={population_per_class}) ...")
    model = SNNClassifier(n_input=n_input, n_hidden=hidden_neurons,
                           n_classes=n_classes,
                           population_per_class=population_per_class,
                           beta=beta, dropout_prob=0.0)  # no dropout at inference
    state_dict = torch.load(os.path.join(fold_dir, "best_model.pt"),
                             map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Running trained SNN forward pass (input -> hidden -> output) ...")
    with torch.no_grad():
        spk_out, mem_out, spk_hidden = model(spikes_in, return_hidden=True)

    trial_class = int(y_test_one[0])
    class_name = CLASS_NAMES.get(trial_class, f"class {trial_class}")
    pred_class = int(model.decode(spk_out)[0].item()) + 1  # back to 1-indexed
    pred_name = CLASS_NAMES.get(pred_class, f"class {pred_class}")
    print(f"  True label: {trial_class} ({class_name})  "
          f"Predicted: {pred_class} ({pred_name})")

    input_spikes = spikes_in[:, 0, :].numpy().T    # (n_input, T)
    hidden_spikes = spk_hidden[:, 0, :].numpy().T  # (n_hidden, T)
    output_spikes = spk_out[:, 0, :].numpy().T     # (n_output, T)
    n_samples = input_spikes.shape[1]
    time_s = np.arange(n_samples) / args.sfreq

    print(f"  Firing rates -- input: {input_spikes.mean():.4f}  "
          f"hidden: {hidden_spikes.mean():.4f}  "
          f"output: {output_spikes.mean():.4f}")

    matplotlib.rcdefaults()
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))
    _raster_panel(axes[0], input_spikes, time_s, "In",
                  "Input Spikes (post-MIBIF)")
    _raster_panel(axes[1], hidden_spikes, time_s, "H",
                  "Hidden Layer Spikes")
    _raster_panel(axes[2], output_spikes, time_s, "Out",
                  "Output Layer Spikes")
    axes[2].set_xticks([time_s[0], time_s[-1]])
    axes[2].set_xticklabels(["n=0", "n=N-1"])
    fig.suptitle(f"Real BNCI2014-001 S{args.subject_id} Trial "
                 f"(true: {class_name}, predicted: {pred_name})", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved: {args.output} (and matching .pdf)")


if __name__ == "__main__":
    main()
