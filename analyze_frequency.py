#!/usr/bin/env python3
"""Frequency-domain diagnostic analysis for MI-EEG.

Produces per-subject and cross-subject visualisations:

  A. Per-class PSD overlay  (motor channels + all-channel mean)
  B. Fisher discriminability spectrum  (4–40 Hz, channel-averaged)
  C. Channel × frequency Fisher ratio heatmap
  D. Time–frequency spectrogram (STFT) per class
  E. Session 1 vs Session 2 PSD shift (cross-session non-stationarity)
  F. Cross-subject Fisher ratio heatmap  (subjects × frequencies)

All figures are saved under ``<results_dir>/FreqAnalysis/Subject_<N>/``.
A summary CSV is written to ``<results_dir>/FreqAnalysis/summary_freq.csv``.

Usage
-----
::

    python analyze_frequency.py --subjects 1 2 3 4 5 6 7 8 9 \\
        --dataset BNCI2014_001 --results-dir Results

    # Skip slow STFT plots:
    python analyze_frequency.py --subjects 1 2 --no-stft

    # Single subject:
    python analyze_frequency.py --subjects 5 --results-dir Results_debug
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, stft

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_frequency")

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

#: BNCI2014_001 channel names (22 channels, 0-indexed)
BNCI2014_001_CH_NAMES: List[str] = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "EOGl",
]

#: Core motor cortex channels highlighted in plots
MOTOR_CH_NAMES: List[str] = ["C3", "C1", "Cz", "C2", "C4"]

#: Extended sensorimotor strip
EXTENDED_MOTOR_CH_NAMES: List[str] = [
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]

#: Class label text (1-indexed) for BNCI2014_001
BNCI2014_001_CLASS_NAMES: Dict[int, str] = {
    1: "Left Hand",
    2: "Right Hand",
    3: "Feet",
    4: "Tongue",
}

CLASS_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]


# ---------------------------------------------------------------------------
# PSD helpers
# ---------------------------------------------------------------------------

def compute_psd_per_class(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Per-class mean and std PSD, averaged over channels.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        1-indexed class labels, shape ``(n_trials,)``.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch segment length (clipped to ``n_samples``).
    fmin, fmax : float
        Frequency limits for the returned slice.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis, shape ``(n_freqs,)``.
    mean_psd : Dict[int, np.ndarray]
        class → mean PSD ``(n_channels, n_freqs)``.
    std_psd : Dict[int, np.ndarray]
        class → std PSD ``(n_channels, n_freqs)``.
    """
    n_trials, n_channels, n_samples = X.shape
    nperseg = min(nperseg, n_samples)

    X_2d = X.reshape(n_trials * n_channels, n_samples)
    freqs, psd_2d = welch(X_2d, fs=sfreq, nperseg=nperseg, axis=-1)
    psd_3d = psd_2d.reshape(n_trials, n_channels, -1)   # (T, C, F)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs     = freqs[freq_mask]
    psd_3d    = psd_3d[:, :, freq_mask]

    mean_psd: Dict[int, np.ndarray] = {}
    std_psd:  Dict[int, np.ndarray] = {}
    for c in np.unique(y):
        mask = y == c
        mean_psd[int(c)] = psd_3d[mask].mean(axis=0)
        std_psd[int(c)]  = psd_3d[mask].std(axis=0)

    return freqs, mean_psd, std_psd


def compute_fisher_channel_freq(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fisher discriminant ratio at every (channel, frequency) pair.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        1-indexed class labels.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch segment length.
    fmin, fmax : float
        Frequency limits.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis, shape ``(n_freqs,)``.
    fisher_mat : np.ndarray
        Fisher ratio matrix, shape ``(n_channels, n_freqs)``.
    """
    n_trials, n_channels, n_samples = X.shape
    nperseg = min(nperseg, n_samples)

    X_2d = X.reshape(n_trials * n_channels, n_samples)
    freqs, psd_2d = welch(X_2d, fs=sfreq, nperseg=nperseg, axis=-1)
    psd_3d = psd_2d.reshape(n_trials, n_channels, -1).astype(np.float64)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs  = freqs[freq_mask]
    psd_3d = psd_3d[:, :, freq_mask]

    classes  = np.unique(y)
    mu_global = psd_3d.mean(axis=0)           # (C, F)
    s_b = np.zeros_like(mu_global)
    s_w = np.zeros_like(mu_global)

    for c in classes:
        mask = y == c
        n_c  = int(mask.sum())
        psd_c = psd_3d[mask]                  # (n_c, C, F)
        mu_c  = psd_c.mean(axis=0)
        s_b  += n_c * (mu_c - mu_global) ** 2
        s_w  += np.sum((psd_c - mu_c) ** 2, axis=0)

    fisher_mat = s_b / (s_w + 1e-30)
    return freqs, fisher_mat


def compute_stft_per_class(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_indices: Optional[List[int]] = None,
    nperseg: int = 64,
    noverlap: int = 56,
    fmin: float = 4.0,
    fmax: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Mean power spectrogram per class (averaged over channels and trials).

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        1-indexed class labels.
    sfreq : float
        Sampling frequency.
    ch_indices : list of int, optional
        Channel subset.  ``None`` → all channels.
    nperseg : int
        STFT window length (64 = 256 ms at 250 Hz).
    noverlap : int
        STFT overlap (56 → step of 8 samples = 32 ms).
    fmin, fmax : float
        Frequency display limits.

    Returns
    -------
    t_stft : np.ndarray
        STFT time axis in seconds.
    freqs_stft : np.ndarray
        STFT frequency axis in Hz.
    power_per_class : Dict[int, np.ndarray]
        class → mean dB-relative power ``(n_freqs, n_times)``.
    """
    if ch_indices is None:
        ch_indices = list(range(X.shape[1]))

    X_sel    = X[:, ch_indices, :]
    classes  = np.unique(y)
    acc_dict: Dict[int, np.ndarray] = {}
    cnt_dict: Dict[int, int]        = {}

    freqs_stft: Optional[np.ndarray] = None
    t_stft:     Optional[np.ndarray] = None

    for c in classes:
        mask = y == c
        X_c  = X_sel[mask]
        for trial in X_c:
            for ch_sig in trial:
                f_s, t_s, Zxx = stft(
                    ch_sig, fs=sfreq,
                    nperseg=nperseg, noverlap=noverlap,
                    boundary=None, padded=False,
                )
                power = np.abs(Zxx) ** 2
                ci = int(c)
                if freqs_stft is None:
                    freqs_stft = f_s
                    t_stft     = t_s
                if ci not in acc_dict:
                    acc_dict[ci] = power.astype(np.float64)
                    cnt_dict[ci] = 1
                else:
                    acc_dict[ci] += power
                    cnt_dict[ci] += 1

    if freqs_stft is None or t_stft is None:
        raise RuntimeError("No STFT data computed.")

    freq_mask  = (freqs_stft >= fmin) & (freqs_stft <= fmax)
    freqs_stft = freqs_stft[freq_mask]

    power_per_class: Dict[int, np.ndarray] = {}
    for ci in acc_dict:
        power_per_class[ci] = (acc_dict[ci] / cnt_dict[ci])[freq_mask, :]

    # Convert to dB relative to cross-class mean (highlights ERD/ERS)
    ref = np.stack(list(power_per_class.values()), axis=0).mean(axis=0)
    eps = ref.mean() * 1e-6
    for ci in power_per_class:
        power_per_class[ci] = 10.0 * np.log10(
            power_per_class[ci] / (ref + eps) + 1e-12
        )

    return t_stft, freqs_stft, power_per_class


# ---------------------------------------------------------------------------
# Channel-name utilities
# ---------------------------------------------------------------------------

def ch_idx_of(ch_names: List[str], targets: List[str]) -> List[int]:
    """Indices of *targets* that appear in *ch_names*."""
    tset = set(targets)
    return [i for i, n in enumerate(ch_names) if n in tset]


def _shade_mi_bands(ax: plt.Axes, semilogy: bool = False) -> None:
    """Add translucent shading for α and β bands and text labels."""
    for lo, hi, lbl, clr in [
        (4,  8,  "θ", "green"),
        (8,  12, "α", "purple"),
        (13, 30, "β", "red"),
    ]:
        ax.axvspan(lo, hi, alpha=0.07, color=clr)
        # x-axis transform: x in data coords, y in axes fraction [0, 1]
        ax.text(
            (lo + hi) / 2, 0.96,
            lbl, ha="center", va="top", fontsize=8, color=clr,
            transform=ax.get_xaxis_transform(),
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_class_psd(
    freqs: np.ndarray,
    mean_psd: Dict[int, np.ndarray],
    std_psd:  Dict[int, np.ndarray],
    ch_names: List[str],
    motor_ch_idx: List[int],
    class_names: Dict[int, str],
    subject_id: int,
    session_label: str,
    out_path: Path,
) -> None:
    """Figure A: per-class PSD overlay (all-channel mean + motor channels)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Subject {subject_id} — Class PSD ({session_label})", fontsize=13
    )

    all_ch_idx = list(range(len(ch_names)))
    panel_groups = [
        (axes[0], all_ch_idx,    "All channels (mean)"),
        (axes[1], motor_ch_idx,  "Motor channels ("
                                  + ", ".join(ch_names[i] for i in motor_ch_idx)
                                  + ")"),
    ]

    for ax, ch_idx, title in panel_groups:
        if not ch_idx:
            ch_idx = all_ch_idx
        for ci, (c, mu_c) in enumerate(sorted(mean_psd.items())):
            sd_c  = std_psd[c]
            mu    = mu_c[ch_idx, :].mean(axis=0)
            sem   = sd_c[ch_idx, :].mean(axis=0) / np.sqrt(len(ch_idx))
            label = class_names.get(c, f"Class {c}")
            clr   = CLASS_COLORS[ci % len(CLASS_COLORS)]
            ax.semilogy(freqs, mu, color=clr, label=label, linewidth=1.6)
            ax.fill_between(freqs, np.maximum(mu - sem, 1e-20), mu + sem,
                            color=clr, alpha=0.15)

        _shade_mi_bands(ax, semilogy=True)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power spectral density (log)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved: %s", out_path.name)


def plot_fisher_spectrum(
    fisher_freqs: np.ndarray,
    fisher_mat:   np.ndarray,
    ch_names: List[str],
    motor_ch_idx: List[int],
    subject_id: int,
    out_path: Path,
) -> None:
    """Figure B: Fisher ratio spectrum — all-channel and motor-only."""
    fisher_all   = fisher_mat.mean(axis=0)
    fisher_motor = (fisher_mat[motor_ch_idx, :].mean(axis=0)
                    if motor_ch_idx else fisher_all)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"Subject {subject_id} — Fisher Discriminability Spectrum (Session 1)",
        fontsize=13,
    )

    ax.plot(fisher_freqs, fisher_all,   color="steelblue",  lw=1.8, label="All channels")
    ax.plot(fisher_freqs, fisher_motor, color="darkorange", lw=1.8, ls="--", label="Motor channels")

    _shade_mi_bands(ax)

    # Mark top-3 peaks
    fmask  = (fisher_freqs >= 4.0) & (fisher_freqs <= 40.0)
    f4_40  = fisher_freqs[fmask]
    v4_40  = fisher_all[fmask]
    peaks  = np.argsort(v4_40)[::-1][:3]
    for rank, pidx in enumerate(peaks):
        ax.axvline(f4_40[pidx], color="crimson", lw=0.9, ls=":", alpha=0.8)
        ax.annotate(
            f"#{rank+1}\n{f4_40[pidx]:.1f} Hz",
            xy=(f4_40[pidx], v4_40[pidx]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color="crimson", va="bottom",
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fisher Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved: %s", out_path.name)


def plot_channel_freq_heatmap(
    fisher_freqs: np.ndarray,
    fisher_mat:   np.ndarray,
    ch_names: List[str],
    motor_ch_names: List[str],
    subject_id: int,
    out_path: Path,
    fmin: float = 4.0,
    fmax: float = 40.0,
) -> None:
    """Figure C: Channel × frequency Fisher ratio heatmap."""
    # Restrict to display range
    fmask = (fisher_freqs >= fmin) & (fisher_freqs <= fmax)
    freqs_plot = fisher_freqs[fmask]

    # Drop EOG channels
    eog_idx = {i for i, n in enumerate(ch_names) if "EOG" in n.upper()}
    eeg_idx = [i for i in range(len(ch_names)) if i not in eog_idx]
    mat     = fisher_mat[eeg_idx, :][:, fmask]
    ch_plot = [ch_names[i] for i in eeg_idx]

    # Sort channels by max Fisher score (most discriminative first)
    order      = np.argsort(mat.max(axis=1))[::-1]
    mat_sorted = mat[order, :]
    ch_sorted  = [ch_plot[i] for i in order]

    # Add "▶" prefix for motor cortex channels
    motor_set  = set(motor_ch_names)
    ch_labels  = [f"▶ {n}" if n in motor_set else n for n in ch_sorted]

    # Freq-axis ticks
    freq_ticks    = np.arange(4, 41, 4)
    freq_tick_pos = [int(np.argmin(np.abs(freqs_plot - f))) for f in freq_ticks]

    fig, ax = plt.subplots(figsize=(12, max(5, len(ch_sorted) // 2)))
    fig.suptitle(
        f"Subject {subject_id} — Channel × Frequency Fisher Heatmap (Session 1)",
        fontsize=13,
    )

    im = ax.imshow(
        mat_sorted,
        aspect="auto", origin="upper",
        cmap="YlOrRd", interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Fisher Ratio")

    ax.set_xticks(freq_tick_pos)
    ax.set_xticklabels([f"{f:.0f}" for f in freq_ticks])
    ax.set_yticks(range(len(ch_labels)))
    ax.set_yticklabels(ch_labels, fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Channel (sorted by peak Fisher, ▶ = motor cortex)")

    # Vertical reference lines for canonical band boundaries
    for f_bnd in (8.0, 12.0, 13.0, 30.0):
        idx = int(np.argmin(np.abs(freqs_plot - f_bnd)))
        ax.axvline(idx, color="white", lw=0.8, ls="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved: %s", out_path.name)


def plot_stft_per_class(
    t_stft:          np.ndarray,
    freqs_stft:      np.ndarray,
    power_per_class: Dict[int, np.ndarray],
    class_names:     Dict[int, str],
    subject_id: int,
    out_path: Path,
) -> None:
    """Figure D: STFT time-frequency spectrogram per class (dB re cross-class mean)."""
    classes = sorted(power_per_class.keys())
    n_cls   = len(classes)
    ncols   = min(n_cls, 2)
    nrows   = int(np.ceil(n_cls / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        f"Subject {subject_id} — STFT Spectrogram per Class "
        f"(Session 1, motor channels, dB re mean)",
        fontsize=13,
    )

    all_v = np.concatenate([p.ravel() for p in power_per_class.values()])
    clim  = float(np.percentile(np.abs(all_v), 98))

    for i, c in enumerate(classes):
        ax    = axes[i // ncols][i % ncols]
        power = power_per_class[c]   # (n_freqs, n_times)
        im    = ax.imshow(
            power,
            aspect="auto", origin="lower",
            extent=[float(t_stft[0]), float(t_stft[-1]),
                    float(freqs_stft[0]), float(freqs_stft[-1])],
            cmap="RdBu_r", vmin=-clim, vmax=clim,
            interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="dB re mean")

        label = class_names.get(c, f"Class {c}")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        # α and β band boundaries
        for f_bnd, lbl in [(8, "α lo"), (12, "α hi"), (13, "β lo"), (30, "β hi")]:
            ax.axhline(f_bnd, color="white", lw=0.8, ls="--", alpha=0.6)
        ax.text(float(t_stft[-1]), 10.0, " α", color="white", va="center", fontsize=8)
        ax.text(float(t_stft[-1]), 21.5, " β", color="white", va="center", fontsize=8)

    for j in range(len(classes), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved: %s", out_path.name)


def plot_session_shift(
    freqs: np.ndarray,
    mean_psd_s1: Dict[int, np.ndarray],
    mean_psd_s2: Dict[int, np.ndarray],
    ch_names: List[str],
    motor_ch_idx: List[int],
    class_names: Dict[int, str],
    subject_id: int,
    out_path: Path,
) -> None:
    """Figure E: Session 1 vs Session 2 PSD and relative shift."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Subject {subject_id} — Session 1 vs Session 2 PSD Shift (motor channels)",
        fontsize=13,
    )

    classes = sorted(set(mean_psd_s1.keys()) & set(mean_psd_s2.keys()))
    ch_idx  = motor_ch_idx if motor_ch_idx else list(range(len(ch_names)))

    for ci, c in enumerate(classes):
        clr   = CLASS_COLORS[ci % len(CLASS_COLORS)]
        label = class_names.get(c, f"Class {c}")

        mu1 = mean_psd_s1[c][ch_idx, :].mean(axis=0)
        mu2 = mean_psd_s2[c][ch_idx, :].mean(axis=0)

        axes[0].semilogy(freqs, mu1, color=clr, lw=1.5, label=label)
        axes[1].semilogy(freqs, mu2, color=clr, lw=1.5, ls="--", label=label)

        eps        = np.abs(mu1).mean() * 1e-6
        shift_pct  = (mu2 - mu1) / (mu1 + eps) * 100.0
        axes[2].plot(freqs, shift_pct, color=clr, lw=1.5, label=label)

    axes[2].axhline(0, color="black", lw=0.8)

    titles = ["Session 1 (train)", "Session 2 (test)", "Δ = (S2−S1)/S1 (%)"]
    for ax, title in zip(axes, titles):
        _shade_mi_bands(ax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Power spectral density (log)")
    axes[2].set_ylabel("Relative change (%)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved: %s", out_path.name)


def plot_cross_subject_heatmap(
    freqs: np.ndarray,
    subject_fisher: Dict[int, np.ndarray],
    out_path: Path,
    fmin: float = 4.0,
    fmax: float = 40.0,
) -> None:
    """Figure F: Subjects × frequencies Fisher ratio heatmap (normalised per subject)."""
    fmask      = (freqs >= fmin) & (freqs <= fmax)
    freqs_plot = freqs[fmask]
    subjects   = sorted(subject_fisher.keys())

    mat = np.stack([subject_fisher[s][fmask] for s in subjects], axis=0)

    # Row-normalise so inter-subject magnitude differences don't dominate
    row_max  = mat.max(axis=1, keepdims=True) + 1e-12
    mat_norm = mat / row_max

    freq_ticks    = np.arange(4, 41, 4)
    freq_tick_pos = [int(np.argmin(np.abs(freqs_plot - f))) for f in freq_ticks]

    fig, ax = plt.subplots(figsize=(14, max(4, len(subjects))))
    fig.suptitle(
        "Cross-Subject Fisher Ratio Heatmap (Session 1, all-channel mean, row-normalised)",
        fontsize=13,
    )

    im = ax.imshow(mat_norm, aspect="auto", origin="upper",
                   cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalised Fisher Ratio")

    ax.set_xticks(freq_tick_pos)
    ax.set_xticklabels([f"{f:.0f}" for f in freq_ticks])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"S{s}" for s in subjects], fontsize=9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Subject")

    for f_bnd in (8.0, 12.0, 13.0, 30.0):
        idx = int(np.argmin(np.abs(freqs_plot - f_bnd)))
        ax.axvline(idx, color="white", lw=0.8, ls="--", alpha=0.6)

    # Annotate peak frequency per subject
    for row, s in enumerate(subjects):
        peak_idx = int(np.argmax(mat_norm[row]))
        ax.text(peak_idx, row, "★", ha="center", va="center",
                fontsize=8, color="black", alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross-subject heatmap: %s", out_path.name)


# ---------------------------------------------------------------------------
# Per-subject runner
# ---------------------------------------------------------------------------

def analyse_subject(
    subject_id: int,
    dataset: str,
    results_dir: Path,
    ch_names: List[str],
    class_names: Dict[int, str],
    motor_ch_names: List[str],
    ext_motor_ch_names: List[str],
    sfreq: float,
    run_stft: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Run all frequency analyses for one subject.

    Returns
    -------
    (fisher_curve, fisher_freqs, summary_row) or ``None`` on failure.
    *fisher_curve* is the channel-averaged Fisher ratio for cross-subject use.
    """
    from fbcsp_snn.datasets import load_moabb

    subj_dir = results_dir / f"Subject_{subject_id}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Subject %d — loading data …", subject_id)

    try:
        X_tr, y_tr, X_te, y_te = load_moabb(dataset, subject_id)
    except Exception as exc:
        logger.error("Subject %d: data loading failed — %s", subject_id, exc)
        return None

    logger.info(
        "  Train: %s  Test: %s  sfreq=%.0f Hz",
        X_tr.shape, X_te.shape, sfreq,
    )

    # Reconcile channel names with actual channel count
    n_ch = X_tr.shape[1]
    ch = list(ch_names)
    if len(ch) > n_ch:
        logger.warning(
            "  ch_names has %d entries, X has %d channels — truncating.",
            len(ch), n_ch,
        )
        ch = ch[:n_ch]
    elif len(ch) < n_ch:
        logger.warning(
            "  ch_names has %d entries, X has %d channels — padding with generic names.",
            len(ch), n_ch,
        )
        ch = ch + [f"CH{i}" for i in range(len(ch), n_ch)]

    motor_idx     = ch_idx_of(ch, motor_ch_names)
    ext_motor_idx = ch_idx_of(ch, ext_motor_ch_names)

    if not motor_idx:
        logger.warning("  No motor channels matched — using all EEG channels.")
        motor_idx     = [i for i in range(n_ch) if "EOG" not in ch[i].upper()]
        ext_motor_idx = motor_idx

    # ------------------------------------------------------------------
    # A — Class PSD overlay
    # ------------------------------------------------------------------
    logger.info("  [A] Class PSD …")
    freqs, mean_tr, std_tr = compute_psd_per_class(X_tr, y_tr, sfreq)
    _,     mean_te, std_te = compute_psd_per_class(X_te, y_te, sfreq)

    plot_class_psd(freqs, mean_tr, std_tr, ch, motor_idx, class_names,
                   subject_id, "Session 1 / Train",
                   subj_dir / "A_class_psd_session1.png")
    plot_class_psd(freqs, mean_te, std_te, ch, motor_idx, class_names,
                   subject_id, "Session 2 / Test",
                   subj_dir / "A_class_psd_session2.png")

    # ------------------------------------------------------------------
    # B + C — Fisher spectrum & channel×freq heatmap
    # ------------------------------------------------------------------
    logger.info("  [B/C] Fisher ratio …")
    fisher_freqs, fisher_mat = compute_fisher_channel_freq(X_tr, y_tr, sfreq)
    fisher_curve_all = fisher_mat.mean(axis=0)   # channel-averaged, for cross-subject

    plot_fisher_spectrum(fisher_freqs, fisher_mat, ch, motor_idx,
                         subject_id, subj_dir / "B_fisher_spectrum.png")
    plot_channel_freq_heatmap(fisher_freqs, fisher_mat, ch, motor_ch_names,
                               subject_id, subj_dir / "C_channel_freq_heatmap.png")

    # ------------------------------------------------------------------
    # D — STFT spectrograms
    # ------------------------------------------------------------------
    if run_stft:
        logger.info("  [D] STFT spectrograms …")
        try:
            t_stft, freqs_stft, power_cls = compute_stft_per_class(
                X_tr, y_tr, sfreq, ch_indices=ext_motor_idx,
            )
            plot_stft_per_class(t_stft, freqs_stft, power_cls,
                                 class_names, subject_id,
                                 subj_dir / "D_stft_per_class.png")
        except Exception as exc:
            logger.warning("  [D] STFT failed: %s", exc)
    else:
        logger.info("  [D] Skipped (--no-stft).")

    # ------------------------------------------------------------------
    # E — Session shift
    # ------------------------------------------------------------------
    logger.info("  [E] Session PSD shift …")
    plot_session_shift(freqs, mean_tr, mean_te, ch, motor_idx, class_names,
                       subject_id, subj_dir / "E_session_shift.png")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    fmask4 = (fisher_freqs >= 4.0) & (fisher_freqs <= 40.0)
    f4     = fisher_freqs[fmask4]
    v4     = fisher_curve_all[fmask4]

    top3_idx  = np.argsort(v4)[::-1][:3]
    top3_freq = f4[top3_idx].tolist()
    top3_val  = v4[top3_idx].tolist()

    # Top-3 channels by max Fisher ratio
    eog_set     = {i for i, n in enumerate(ch) if "EOG" in n.upper()}
    ch_max      = [(fisher_mat[i, fmask4].max(), ch[i])
                   for i in range(len(ch)) if i not in eog_set]
    ch_max_sort = sorted(ch_max, reverse=True)[:3]

    # Band-specific Fisher power
    alpha_mask = (fisher_freqs >= 8.0)  & (fisher_freqs <= 12.0)
    beta_mask  = (fisher_freqs >= 13.0) & (fisher_freqs <= 30.0)
    alpha_fval = float(fisher_curve_all[alpha_mask].mean()) if alpha_mask.any() else 0.0
    beta_fval  = float(fisher_curve_all[beta_mask].mean())  if beta_mask.any()  else 0.0

    summary_row = {
        "subject": subject_id,
        "top_freq_1": f"{top3_freq[0]:.1f}" if len(top3_freq) > 0 else "",
        "top_freq_2": f"{top3_freq[1]:.1f}" if len(top3_freq) > 1 else "",
        "top_freq_3": f"{top3_freq[2]:.1f}" if len(top3_freq) > 2 else "",
        "top_fisher_1": f"{top3_val[0]:.4f}" if len(top3_val) > 0 else "",
        "top_fisher_2": f"{top3_val[1]:.4f}" if len(top3_val) > 1 else "",
        "top_fisher_3": f"{top3_val[2]:.4f}" if len(top3_val) > 2 else "",
        "top_channel_1": ch_max_sort[0][1] if len(ch_max_sort) > 0 else "",
        "top_channel_2": ch_max_sort[1][1] if len(ch_max_sort) > 1 else "",
        "top_channel_3": ch_max_sort[2][1] if len(ch_max_sort) > 2 else "",
        "top_channel_fisher_1": f"{ch_max_sort[0][0]:.4f}" if len(ch_max_sort) > 0 else "",
        "top_channel_fisher_2": f"{ch_max_sort[1][0]:.4f}" if len(ch_max_sort) > 1 else "",
        "top_channel_fisher_3": f"{ch_max_sort[2][0]:.4f}" if len(ch_max_sort) > 2 else "",
        "alpha_fisher": f"{alpha_fval:.4f}",
        "beta_fisher":  f"{beta_fval:.4f}",
    }

    logger.info(
        "  S%d summary: peaks @ %s Hz | top ch: %s | α=%.3f β=%.3f",
        subject_id,
        [f"{f:.1f}" for f in top3_freq],
        [n for _, n in ch_max_sort],
        alpha_fval, beta_fval,
    )

    return fisher_curve_all, fisher_freqs, summary_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frequency-domain diagnostic analysis for MI-EEG"
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(range(1, 10)),
        help="Subject IDs to analyse (default: 1–9).",
    )
    parser.add_argument(
        "--dataset", default="BNCI2014_001",
        help="MOABB dataset name (default: BNCI2014_001).",
    )
    parser.add_argument(
        "--results-dir", default="Results",
        help="Root output directory (default: Results).",
    )
    parser.add_argument(
        "--no-stft", dest="run_stft", action="store_false", default=True,
        help="Skip STFT spectrogram plots (faster runtime).",
    )
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "FreqAnalysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset

    # Dataset-specific metadata
    if dataset == "BNCI2014_001":
        ch_names          = BNCI2014_001_CH_NAMES
        class_names       = BNCI2014_001_CLASS_NAMES
        motor_ch_names    = MOTOR_CH_NAMES
        ext_motor_names   = EXTENDED_MOTOR_CH_NAMES
        sfreq             = 250.0
    else:
        from fbcsp_snn.datasets import DATASET_REGISTRY
        info          = DATASET_REGISTRY.get(dataset, {})
        n_ch_reg      = int(info.get("n_channels", 64))
        ch_names      = [f"CH{i}" for i in range(n_ch_reg)]
        class_names   = {}
        motor_ch_names  = []
        ext_motor_names = []
        sfreq           = float(info.get("sfreq", 250))
        logger.warning(
            "No channel metadata for '%s' — generic names used.", dataset
        )

    # Per-subject loop
    subject_fisher:  Dict[int, np.ndarray] = {}
    subject_freqs:   Dict[int, np.ndarray] = {}   # store actual freq axis per subject
    summary_rows:    Dict[int, Dict]       = {}

    for subj in args.subjects:
        result = analyse_subject(
            subject_id=subj,
            dataset=dataset,
            results_dir=out_dir,
            ch_names=ch_names,
            class_names=class_names,
            motor_ch_names=motor_ch_names,
            ext_motor_ch_names=ext_motor_names,
            sfreq=sfreq,
            run_stft=args.run_stft,
        )
        if result is not None:
            fisher_curve, fisher_freqs, summary_row = result
            subject_fisher[subj]  = fisher_curve
            subject_freqs[subj]   = fisher_freqs
            summary_rows[subj]    = summary_row

    # Cross-subject Fisher heatmap (Figure F)
    if len(subject_fisher) >= 2:
        # All subjects share the same freq axis (same dataset/sfreq/nperseg)
        ref_freqs = subject_freqs[min(subject_freqs.keys())]
        plot_cross_subject_heatmap(
            ref_freqs,
            subject_fisher,
            out_path=out_dir / "F_cross_subject_fisher_heatmap.png",
        )
    else:
        logger.info("Fewer than 2 subjects — skipping cross-subject heatmap.")

    # Summary CSV
    csv_path = out_dir / "summary_freq.csv"
    fieldnames = [
        "subject",
        "top_freq_1", "top_freq_2", "top_freq_3",
        "top_fisher_1", "top_fisher_2", "top_fisher_3",
        "top_channel_1", "top_channel_2", "top_channel_3",
        "top_channel_fisher_1", "top_channel_fisher_2", "top_channel_fisher_3",
        "alpha_fisher", "beta_fisher",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subj in sorted(summary_rows.keys()):
            writer.writerow(summary_rows[subj])
    logger.info("Summary CSV: %s", csv_path)

    # Console table
    print("\n" + "=" * 72)
    print(f"  Frequency Analysis Summary — {dataset}")
    print("=" * 72)
    print(f"  {'Subj':<6} {'Peak1 Hz':>8} {'Peak2 Hz':>8} {'Peak3 Hz':>8}"
          f"  {'Top ch1':>7}  {'Top ch2':>7}  {'α Fisher':>9}  {'β Fisher':>9}")
    print("  " + "-" * 68)
    for subj in sorted(summary_rows.keys()):
        r = summary_rows[subj]
        print(
            f"  S{r['subject']:<5} {r['top_freq_1']:>8} {r['top_freq_2']:>8} "
            f"{r['top_freq_3']:>8}  {r['top_channel_1']:>7}  {r['top_channel_2']:>7}"
            f"  {r['alpha_fisher']:>9}  {r['beta_fisher']:>9}"
        )
    print("=" * 72)
    print(f"\nFigures saved in: {out_dir}\n")


if __name__ == "__main__":
    main()
