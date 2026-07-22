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

# fbcsp_snn.datasets.load_moabb builds label_map from sorted(set(y_all)) --
# i.e. ALPHABETICAL by class name, not the "standard" BCI-IV-2a paper
# convention (1=left,2=right,3=feet,4=tongue). Verified from a real
# load_moabb log: label_map: {feet: 1, left_hand: 2, right_hand: 3, tongue: 4}.
CLASS_NAMES = {1: "Feet", 2: "Left Hand", 3: "Right Hand", 4: "Tongue"}


def _concat_projections(proj: dict) -> np.ndarray:
    """Matches fbcsp_snn/pipeline.py's _concat_projections exactly."""
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def _raster_panel(ax, spikes_1d_by_feat: np.ndarray, time_s: np.ndarray,
                   ylabel: str) -> None:
    """Draw a full spike raster -- EVERY channel/neuron, one dot per spike.

    Unlike the synthetic pipeline-stage figures (which abbreviate to
    first-3/separator/last-3 so the figure stays legible and N-independent),
    this plot needs to show the real scale of the trained network: all
    input features, all hidden neurons, all output neurons. A dot-per-spike
    scatter (matching the style of the existing spike_propagation.png
    diagnostic already produced by fbcsp_snn/visualization.py) shows the
    genuine spike density/pattern across the whole population without
    needing per-row labels.

    Parameters
    ----------
    spikes_1d_by_feat : np.ndarray
        Shape (n_features, n_timesteps), binary.
    """
    n_feat, n_t = spikes_1d_by_feat.shape
    neuron_idx, t_idx = np.where(spikes_1d_by_feat > 0)
    ax.scatter(time_s[t_idx], neuron_idx, s=10, marker="s", color="black",
               linewidths=0)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_ylim(-1, n_feat)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis="both", labelsize=20)


def _output_raster_panel(ax, spikes_1d_by_feat: np.ndarray, time_s: np.ndarray,
                          n_classes: int, population_per_class: int,
                          class_names: dict) -> None:
    """Full spike raster for the output layer, grouped into its class
    populations: a dashed separator between each population block, and
    y-tick labels showing the actual class name (not a raw neuron index) at
    the centre of each block. Population block k (0-indexed) corresponds to
    1-indexed class k+1 -- see the CLASS_NAMES note above for why that
    mapping is alphabetical, not the "standard" BCI-IV-2a paper convention.
    """
    n_feat, n_t = spikes_1d_by_feat.shape
    neuron_idx, t_idx = np.where(spikes_1d_by_feat > 0)
    ax.scatter(time_s[t_idx], neuron_idx, s=10, marker="s", color="black",
               linewidths=0)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_ylim(-1, n_feat)

    for k in range(1, n_classes):
        ax.axhline(k * population_per_class - 0.5, color="0.5",
                   linestyle="--", linewidth=1)

    centers = [(k + 0.5) * population_per_class for k in range(n_classes)]
    labels = [class_names.get(k + 1, f"class {k + 1}") for k in range(n_classes)]
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=22)
    ax.set_ylabel("Output", fontsize=24)
    ax.tick_params(axis="x", labelsize=20)


def _decision_panel(ax, spikes_1d_by_feat: np.ndarray, time_s: np.ndarray,
                     n_classes: int, population_per_class: int,
                     class_names: dict, true_class_1idx: int,
                     window: int = 40) -> None:
    """Plot the leading population using a SLIDING WINDOW of spike counts
    (default 40 timesteps ~= 160 ms at 250 Hz) -- a local measure, not
    accumulated from t=0. Purely instantaneous (single-timestep) counts
    are too noisy to read (each population only has 20 neurons, so a
    single timestep's count is a small, volatile integer); a full
    from-t=0 cumulative trace was ruled out because the reported accuracy
    elsewhere in the paper is a single total-count argmax evaluated once
    over the whole trial (SNNClassifier.decode()), and a running-from-t=0
    trace would visually imply an evolving decision that doesn't match
    that logic. A short sliding window is a middle ground: still a local
    "what's active right now" measure, just smoothed enough to be
    readable. A dotted red line marks the true class for direct
    visual comparison.
    """
    n_feat, n_t = spikes_1d_by_feat.shape
    per_class = spikes_1d_by_feat.reshape(n_classes, population_per_class, n_t)
    class_counts = per_class.sum(axis=1)   # (n_classes, T) -- count AT each t, no accumulation
    cs = np.cumsum(class_counts, axis=1)
    cs = np.concatenate([np.zeros((n_classes, 1)), cs], axis=1)  # prefix sum, leading zero
    win_start = np.maximum(0, np.arange(n_t) - window + 1)
    windowed = cs[:, np.arange(n_t) + 1] - cs[:, win_start]  # (n_classes, T)
    leading = np.argmax(windowed, axis=0)  # (T,), 0-indexed class

    ax.step(time_s, leading, where="post", color="black", linewidth=1.3)
    ax.axhline(true_class_1idx - 1, color="red", linestyle=":",
               linewidth=1.2, alpha=0.7)
    ax.set_ylim(-0.5, n_classes - 0.5)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([class_names.get(k + 1, f"class {k + 1}")
                         for k in range(n_classes)], fontsize=22)
    ax.set_ylabel("Decision", fontsize=24)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.tick_params(axis="x", labelsize=20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="Results_snn_raster_demo")
    ap.add_argument("--subject-id", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--trial-idx", type=int, default=0,
                     help="Index into the chosen split's trials to plot.")
    ap.add_argument("--split", choices=["test", "train"], default="test",
                     help="Which session to draw the trial from. 'test' "
                          "(default) is the held-out session the model "
                          "never trained on -- uses save_test_spikes.py's "
                          "precomputed spikes when available. 'train' is "
                          "the training session the model DID see; there "
                          "is no precomputed cache for it, so it's always "
                          "derived fresh through the same fitted "
                          "filter bank -> CSP -> z-norm -> encoder -> MIBIF "
                          "chain.")
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

    trial_idx = args.trial_idx
    precomputed_spikes_path = os.path.join(fold_dir, "test_spikes.pt")
    precomputed_labels_path = os.path.join(fold_dir, "test_labels.npy")
    use_precomputed = (args.split == "test"
                        and os.path.exists(precomputed_spikes_path)
                        and os.path.exists(precomputed_labels_path))

    if use_precomputed:
        # save_test_spikes.py already ran the real filter bank -> fitted CSP
        # -> fitted z-norm -> encoder -> MIBIF chain for the WHOLE test set
        # as part of this same training job -- reuse that directly instead
        # of re-deriving it (faster, and removes any risk of a subtle
        # mismatch from recomputing it ourselves).
        print(f"Loading precomputed test spikes from {precomputed_spikes_path} ...")
        all_spikes = torch.load(precomputed_spikes_path, map_location="cpu")  # (T, n_test, n_feat)
        all_labels_0idx = np.load(precomputed_labels_path)  # 0-indexed
        n_trials_loaded, n_feat_loaded = all_spikes.shape[1], all_spikes.shape[2]
        # .clone() is essential here: a plain slice is a VIEW into the full
        # 220MB tensor's storage, so `del all_spikes` below would NOT
        # actually free that memory as long as spikes_in still references
        # it -- this machine is memory-constrained enough that keeping the
        # full tensor alive causes OOM/segfaults during the forward pass.
        spikes_in = all_spikes[:, trial_idx:trial_idx + 1, :].clone()
        y_trial_one = np.array([all_labels_0idx[trial_idx] + 1])  # back to 1-indexed
        del all_spikes, all_labels_0idx
        gc.collect()
        print(f"  Loaded {n_trials_loaded} precomputed test trials, "
              f"{n_feat_loaded} features -- using trial {trial_idx}")
    else:
        # No precomputed cache for the training split (save_test_spikes.py
        # only ever saves the TEST session), and this is also the fallback
        # for the test split when that cache is absent -- either way,
        # derive fresh through the exact same fitted chain.
        print(f"Deriving from real MOABB data ({args.split} session): "
              f"BNCI2014_001, subject {args.subject_id} ...")
        X_train, y_train, X_test, y_test = load_moabb(params["dataset"], args.subject_id)
        if args.split == "train":
            X_split, y_split = X_train, y_train
            del X_test, y_test
        else:
            X_split, y_split = X_test, y_test
            del X_train, y_train
        y_trial_one = y_split[trial_idx:trial_idx + 1]
        X_trial_one = X_split[trial_idx:trial_idx + 1]
        del X_split, y_split
        gc.collect()

        print(f"Applying real filter bank + fitted CSP + fitted z-norm "
              f"(training-fold-fitted parameters, {args.split} trial, "
              f"no leakage since only the parameters were fit on training "
              f"data -- this trial itself is just being transformed) ...")
        Xb_trial_one = apply_filter_bank(X_trial_one, bands, args.sfreq)
        proj_trial_one = csp.transform(Xb_trial_one)
        X_concat_trial_one = _concat_projections(proj_trial_one).astype(np.float32)
        X_norm_test = znorm.transform(X_concat_trial_one)  # (1, n_feat, n_samples)
        del Xb_trial_one, proj_trial_one, X_concat_trial_one
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

    trial_class = int(y_trial_one[0])
    class_name = CLASS_NAMES.get(trial_class, f"class {trial_class}")
    pred_class = int(model.decode(spk_out)[0].item()) + 1  # back to 1-indexed
    pred_name = CLASS_NAMES.get(pred_class, f"class {pred_class}")
    print(f"  [{args.split} split, trial {trial_idx}]  True label: "
          f"{trial_class} ({class_name})  Predicted: {pred_class} ({pred_name})")

    input_spikes = spikes_in[:, 0, :].numpy().T    # (n_input, T)
    hidden_spikes = spk_hidden[:, 0, :].numpy().T  # (n_hidden, T)
    output_spikes = spk_out[:, 0, :].numpy().T     # (n_output, T)
    n_samples = input_spikes.shape[1]
    time_s = np.arange(n_samples) / args.sfreq

    print(f"  Firing rates -- input: {input_spikes.mean():.4f}  "
          f"hidden: {hidden_spikes.mean():.4f}  "
          f"output: {output_spikes.mean():.4f}")

    matplotlib.rcdefaults()
    fig, axes = plt.subplots(
        4, 1, figsize=(13, 16),
        gridspec_kw={"height_ratios": [3, 1.5, 2.2, 1.5]},
        constrained_layout=True,
    )
    _raster_panel(axes[0], input_spikes, time_s, "Input features")
    _raster_panel(axes[1], hidden_spikes, time_s, "Hidden neurons")
    _output_raster_panel(axes[2], output_spikes, time_s, n_classes,
                          population_per_class, CLASS_NAMES)
    _decision_panel(axes[3], output_spikes, time_s, n_classes,
                    population_per_class, CLASS_NAMES, trial_class)
    for ax in axes[:3]:
        ax.set_xticklabels([])
    axes[3].set_xlabel("Time (s)", fontsize=24)

    # Each panel's y-tick labels have different widths (numeric ticks for
    # input/hidden vs. class-name text for output/decision), so the
    # ylabels would otherwise sit at different distances from the axes.
    # align_ylabels lines them all up on one common vertical line.
    fig.align_ylabels(axes)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved: {args.output} (and matching .pdf)")


if __name__ == "__main__":
    main()
