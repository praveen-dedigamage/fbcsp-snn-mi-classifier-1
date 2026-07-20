"""Plot how one real EEG trial changes through each pipeline stage.

Generates the "signal transformation" figure requested in the paper's
TODO.md item #1: raw -> filtered -> CSP-projected -> z-normalised ->
spike-encoded, for one representative trial, using the real pipeline
code (not synthetic data or a schematic illustration).

Shows multiple channels/features (default 6) at every stage, stacked with
a vertical offset like a standard EEG montage plot -- a single channel
makes CSP look like a per-channel filter instead of the spatial
combination across all channels that it actually is.

Must be run where the real dependencies and data exist (Puhti) -- needs
torch, scipy, MOABB/MNE, and network access to fetch/cache the dataset on
first run.

Usage (from the repo root):
    python plot_pipeline_signals.py --subject-id 1 --trial-idx 0
    python plot_pipeline_signals.py --subject-id 1 --moabb-dataset BNCI2014_001 \
        --band-idx 1 --n-lines 6 --output figures/pipeline_signals_S1.png
"""
from __future__ import annotations

import argparse
import os

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

# Preferred channel names if real montage info is available (standard
# sensorimotor-cortex sites for MI, checked against the real channel list
# rather than assumed -- see _get_channel_names).
MOTOR_CHANNEL_PREFERENCE = [
    "C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "C1", "C2", "C5", "C6",
]


def _concat_projections(proj: dict) -> np.ndarray:
    """Concatenate CSP projections from all pairs along the feature axis.

    Mirrors ``fbcsp_snn.pipeline._concat_projections`` exactly (duplicated
    here rather than imported, since that helper is private to the
    pipeline module and this script has no other dependency on it).
    """
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def _get_channel_names(dataset_name: str, subject_id: int) -> list | None:
    """Best-effort fetch of the REAL channel names via MOABB's dataset object.

    ``load_moabb`` discards channel names (returns plain numpy arrays via
    ``paradigm.get_data(..., return_epochs=False)``), so this queries the
    underlying MOABB dataset directly. Returns ``None`` (never a guessed
    list) if anything about this fails, so callers always know whether the
    names are real or a fallback -- never silently mislabel a channel.
    """
    try:
        import moabb.datasets as moabb_ds

        info = DATASET_REGISTRY[dataset_name]
        DatasetCls = getattr(moabb_ds, info["moabb_cls"])
        dataset = DatasetCls(**info.get("dataset_kwargs", {}))
        data = dataset.get_data(subjects=[subject_id])
        subj_data = data[subject_id]
        first_session = next(iter(subj_data.values()))
        first_run = next(iter(first_session.values()))
        return list(first_run.info["ch_names"])
    except Exception as exc:  # noqa: BLE001 -- best-effort, must not crash the plot
        print(f"WARNING: could not fetch real channel names ({exc}); "
              f"falling back to plain channel indices ch0, ch1, ...")
        return None


def _select_channels(ch_names: list | None, n_channels: int,
                      n_available: int) -> tuple:
    """Pick channel indices + labels: prefer real motor-cortex channels.

    Falls back to the first *n_channels* indices (labelled ``ch<i>``) if
    channel names are unavailable, or if fewer than *n_channels* of the
    preferred motor sites are actually present in this montage.
    """
    n_channels = min(n_channels, n_available)
    if ch_names is not None:
        upper = [c.upper() for c in ch_names]
        preferred_idx = [upper.index(name) for name in
                          (n.upper() for n in MOTOR_CHANNEL_PREFERENCE)
                          if name.upper() in upper]
        if len(preferred_idx) >= n_channels:
            idx = preferred_idx[:n_channels]
            return idx, [ch_names[i] for i in idx]
        # Not enough named motor channels found -- fall back to first N,
        # but still use the real names since we have them.
        idx = list(range(n_channels))
        return idx, [ch_names[i] for i in idx]
    idx = list(range(n_channels))
    return idx, [f"ch{i}" for i in idx]


def _stacked_plot(ax, time_s, traces, labels, colors=None):
    """Plot multiple 1-D traces on one axis, vertically offset (EEG-montage style)."""
    n = len(traces)
    # Offset by ~3x the max abs amplitude across all traces, for even spacing.
    scale = max(np.max(np.abs(t)) for t in traces) if traces else 1.0
    offset_step = scale * 2.2
    for i, (trace, label) in enumerate(zip(traces, labels)):
        offset = (n - 1 - i) * offset_step
        color = colors[i] if colors else None
        ax.plot(time_s, trace + offset, linewidth=0.7, color=color)
        ax.text(-0.02, offset, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8)
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject-id", type=int, default=1)
    ap.add_argument("--moabb-dataset", default="BNCI2014_001")
    ap.add_argument("--trial-idx", type=int, default=0,
                     help="Index into the training split to visualise.")
    ap.add_argument("--band-idx", type=int, default=1,
                     help="Which of the 6 fixed bands to show as 'filtered' "
                          "(default 1 = 8-14 Hz mu/alpha band).")
    ap.add_argument("--n-lines", type=int, default=6,
                     help="Number of channels (raw/filtered panels) and CSP "
                          "features (CSP/z-norm/spike panels) to show, "
                          "stacked. Default 6.")
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

    ch_names = _get_channel_names(args.moabb_dataset, args.subject_id)
    ch_idx, ch_labels = _select_channels(ch_names, args.n_lines, X_train.shape[1])
    print(f"  Channels used: {ch_labels}"
          f"{' (real montage names)' if ch_names is not None else ' (fallback indices)'}")

    # ---- Stage 1: raw ----
    raw_traces = [X_train[args.trial_idx, c, :] for c in ch_idx]

    # ---- Stage 2: fixed six-band filter bank (fit CSP on all 6, matching
    # the real pipeline exactly; only one band is plotted for clarity) ----
    print("Applying fixed six-band filter bank ...")
    X_bands = apply_filter_bank(X_train, FREQ_BANDS, sfreq, order=4,
                                 filter_type="butterworth")
    filtered_traces = [X_bands[args.band_idx][args.trial_idx, c, :] for c in ch_idx]

    # ---- Stage 3: dual-end pairwise CSP (real defaults) ----
    print("Fitting dual-end pairwise CSP ...")
    csp = PairwiseCSP(m=args.m, lambda_r=0.0001,
                       euclidean_alignment=True, riemannian_mean=True,
                       dual_end=True)
    csp.fit(X_bands, y_train)
    proj = csp.transform(X_bands)
    X_concat = _concat_projections(proj)  # (n_trials, n_features, n_samples)
    print(f"  CSP-projected: {X_concat.shape}")
    n_feat = min(args.n_lines, X_concat.shape[1])
    feat_idx = list(range(n_feat))
    feat_labels = [f"CSP feat. {i}" for i in feat_idx]
    csp_traces = [X_concat[args.trial_idx, f, :] for f in feat_idx]

    # ---- Stage 4: z-normalisation ----
    znorm = ZNormaliser()
    X_norm = znorm.fit_transform(X_concat)
    znorm_traces = [X_norm[args.trial_idx, f, :] for f in feat_idx]

    # ---- Stage 5: adaptive-threshold spike encoding (real defaults) ----
    print("Encoding spikes (adaptive-threshold delta encoder) ...")
    t = torch.from_numpy(X_norm.astype(np.float32)).permute(2, 0, 1)  # (T, B, F)
    spikes = encode_tensor(t, base_thresh=0.001, adapt_inc=0.6, decay=0.95,
                            encoder_type="delta")
    spike_traces = [spikes[:, args.trial_idx, f].cpu().numpy() for f in feat_idx]

    # ---- Plot ----
    n_samples = X_train.shape[2]
    time_s = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)

    _stacked_plot(axes[0], time_s, raw_traces, ch_labels)
    axes[0].set_title(f"1. Raw EEG ({len(ch_idx)} channels)")

    lo, hi = FREQ_BANDS[args.band_idx]
    _stacked_plot(axes[1], time_s, filtered_traces, ch_labels)
    axes[1].set_title(f"2. After causal bandpass filter ({lo}-{hi} Hz band)")

    _stacked_plot(axes[2], time_s, csp_traces, feat_labels)
    axes[2].set_title(f"3. After dual-end pairwise CSP "
                       f"({n_feat} spatial-filter outputs)")

    _stacked_plot(axes[3], time_s, znorm_traces, feat_labels)
    axes[3].set_title("4. After z-normalisation")

    # Proper multi-row spike raster: one row per feature.
    for i, spk in enumerate(spike_traces):
        row = n_feat - 1 - i
        spike_times = time_s[spk > 0.5]
        axes[4].vlines(spike_times, ymin=row + 0.1, ymax=row + 0.9,
                        color="black", linewidth=0.6)
    axes[4].set_ylim(0, n_feat)
    axes[4].set_yticks([n_feat - 1 - i + 0.5 for i in range(n_feat)])
    axes[4].set_yticklabels(feat_labels, fontsize=8)
    axes[4].set_title("5. After adaptive-threshold spike encoding")
    axes[4].set_xlabel("Time (s)")

    fig.suptitle(f"{args.moabb_dataset} Subject {args.subject_id}, "
                 f"trial {args.trial_idx} (class {int(y_train[args.trial_idx])}): "
                 f"signal through the pipeline", fontsize=11)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])

    output = args.output or f"figures/pipeline_signals_S{args.subject_id}.png"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved: {output} (and matching .pdf)")


if __name__ == "__main__":
    main()
