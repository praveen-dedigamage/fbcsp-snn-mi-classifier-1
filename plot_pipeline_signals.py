"""Plot how one real EEG trial changes through each pipeline stage.

Generates the "signal transformation" figure requested in the paper's
TODO.md item #1: raw -> filtered -> CSP-projected -> z-normalised ->
spike-encoded, for one representative trial, using the real pipeline
code (not synthetic data or a schematic illustration).

Must be run where the real dependencies and data exist (Puhti) -- needs
torch, scipy, MOABB/MNE, and network access to fetch/cache the dataset on
first run.

Usage (from the repo root):
    python plot_pipeline_signals.py --subject-id 1 --trial-idx 0
    python plot_pipeline_signals.py --subject-id 1 --moabb-dataset BNCI2014_001 \
        --band-idx 1 --channel-idx 7 --pair-feature-idx 0 \
        --output figures/pipeline_signals_S1.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fbcsp_snn.datasets import DATASET_REGISTRY, load_moabb
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank

# Fixed six-band filter bank -- matches config.py's default exactly.
FREQ_BANDS = [(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)]


def _concat_projections(proj: dict) -> np.ndarray:
    """Concatenate CSP projections from all pairs along the feature axis.

    Mirrors ``fbcsp_snn.pipeline._concat_projections`` exactly (duplicated
    here rather than imported, since that helper is private to the
    pipeline module and this script has no other dependency on it).
    """
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject-id", type=int, default=1)
    ap.add_argument("--moabb-dataset", default="BNCI2014_001")
    ap.add_argument("--trial-idx", type=int, default=0,
                     help="Index into the training split to visualise.")
    ap.add_argument("--band-idx", type=int, default=1,
                     help="Which of the 6 fixed bands to show as 'filtered' "
                          "(default 1 = 8-14 Hz mu/alpha band).")
    ap.add_argument("--channel-idx", type=int, default=0,
                     help="Raw EEG channel index to plot.")
    ap.add_argument("--pair-feature-idx", type=int, default=0,
                     help="Index into the concatenated CSP feature axis "
                          "(0 = first spatial filter of the first class "
                          "pair) to trace through CSP/z-norm/encoding.")
    ap.add_argument("--m", type=int, default=4,
                     help="CSP eigenvectors per end (matches config.py "
                          "default: csp_components_per_band=8 -> m=4).")
    ap.add_argument("--output", default=None,
                     help="Output path. Defaults to "
                          "figures/pipeline_signals_S<subject>.png")
    args = ap.parse_args()

    sfreq = float(DATASET_REGISTRY[args.moabb_dataset]["sfreq"])

    print(f"Loading {args.moabb_dataset} subject {args.subject_id} ...")
    X_train, y_train, _, _ = load_moabb(args.moabb_dataset, args.subject_id)
    print(f"  X_train: {X_train.shape}  (n_trials, n_channels, n_samples)")

    # ---- Stage 1: raw ----
    raw_trace = X_train[args.trial_idx, args.channel_idx, :]

    # ---- Stage 2: fixed six-band filter bank (fit CSP on all 6, matching
    # the real pipeline exactly; only one band is plotted for clarity) ----
    print("Applying fixed six-band filter bank ...")
    X_bands = apply_filter_bank(X_train, FREQ_BANDS, sfreq, order=4,
                                 filter_type="butterworth")
    filtered_trace = X_bands[args.band_idx][args.trial_idx, args.channel_idx, :]

    # ---- Stage 3: dual-end pairwise CSP (real defaults) ----
    print("Fitting dual-end pairwise CSP ...")
    csp = PairwiseCSP(m=args.m, lambda_r=0.0001,
                       euclidean_alignment=True, riemannian_mean=True,
                       dual_end=True)
    csp.fit(X_bands, y_train)
    proj = csp.transform(X_bands)
    X_concat = _concat_projections(proj)  # (n_trials, n_features, n_samples)
    print(f"  CSP-projected: {X_concat.shape}")
    csp_trace = X_concat[args.trial_idx, args.pair_feature_idx, :]

    # ---- Stage 4: z-normalisation ----
    znorm = ZNormaliser()
    X_norm = znorm.fit_transform(X_concat)
    znorm_trace = X_norm[args.trial_idx, args.pair_feature_idx, :]

    # ---- Stage 5: adaptive-threshold spike encoding (real defaults) ----
    print("Encoding spikes (adaptive-threshold delta encoder) ...")
    t = torch.from_numpy(X_norm.astype(np.float32)).permute(2, 0, 1)  # (T, B, F)
    spikes = encode_tensor(t, base_thresh=0.001, adapt_inc=0.6, decay=0.95,
                            encoder_type="delta")
    spike_trace = spikes[:, args.trial_idx, args.pair_feature_idx].cpu().numpy()

    # ---- Plot ----
    n_samples = raw_trace.shape[0]
    time_s = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(time_s, raw_trace, color="tab:blue", linewidth=0.8)
    axes[0].set_title(f"1. Raw EEG (channel {args.channel_idx})")
    axes[0].set_ylabel("Amplitude")

    lo, hi = FREQ_BANDS[args.band_idx]
    axes[1].plot(time_s, filtered_trace, color="tab:orange", linewidth=0.8)
    axes[1].set_title(f"2. After causal bandpass filter ({lo}-{hi} Hz band)")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(time_s, csp_trace, color="tab:green", linewidth=0.8)
    axes[2].set_title(f"3. After dual-end pairwise CSP "
                       f"(feature {args.pair_feature_idx})")
    axes[2].set_ylabel("Projected value")

    axes[3].plot(time_s, znorm_trace, color="tab:red", linewidth=0.8)
    axes[3].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[3].set_title("4. After z-normalisation")
    axes[3].set_ylabel("z-score")

    axes[4].vlines(time_s[spike_trace > 0.5], ymin=0, ymax=1,
                   color="black", linewidth=0.6)
    axes[4].set_title("5. After adaptive-threshold spike encoding")
    axes[4].set_ylabel("Spike")
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_yticks([0, 1])
    axes[4].set_xlabel("Time (s)")

    fig.suptitle(f"{args.moabb_dataset} Subject {args.subject_id}, "
                 f"trial {args.trial_idx} (class {int(y_train[args.trial_idx])}): "
                 f"signal through the pipeline", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output = args.output or f"figures/pipeline_signals_S{args.subject_id}.png"
    import os
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved: {output} (and matching .pdf)")


if __name__ == "__main__":
    main()
