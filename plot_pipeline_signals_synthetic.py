"""Plot how a SYNTHETIC EEG trial changes through each pipeline stage.

Unlike ``plot_pipeline_signals.py`` (which needs real MOABB data + torch,
and must run on Puhti), this script is fully self-contained -- only
numpy/scipy/matplotlib, no dependency on the fbcsp_snn package or real
data. It reimplements the same real equations/hyperparameters (fixed
six-band filter bank, dual-end CSP via the generalised eigenvalue
problem, z-normalisation, the adaptive-threshold delta encoder) directly,
so the *mechanism* shown is faithful to the real pipeline even though the
input signal is fabricated -- meant for illustration/explanation, not as
evidence for the paper (use the real-data script for that).

Synthetic signal design: two classes, each trial = background 1/f-ish
noise + white noise on every channel, plus an extra ~10 Hz (mu-band)
oscillation concentrated in a different channel subset per class -- this
gives CSP genuine class-discriminative structure to find, similar in
spirit to real ERD/ERS but not derived from any real recording.

Usage (runs anywhere with numpy/scipy/matplotlib, e.g. locally):
    python plot_pipeline_signals_synthetic.py
    python plot_pipeline_signals_synthetic.py --n-lines 6 --band-idx 1 \
        --output figures/pipeline_signals_synthetic.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt

# Fixed six-band filter bank -- matches config.py's real default exactly.
FREQ_BANDS = [(4, 8), (8, 14), (12, 18), (16, 24), (20, 30), (26, 40)]

# Real encoder defaults (fbcsp_snn/encoding.py / config.py).
BASE_THRESH = 0.001
ADAPT_INC = 0.6
DECAY = 0.95

# Real CSP defaults (fbcsp_snn/preprocessing.py / config.py).
LAMBDA_R = 0.0001


def make_synthetic_trials(n_trials_per_class: int, n_channels: int,
                           n_samples: int, sfreq: float,
                           seed: int) -> tuple:
    """Generate synthetic 2-class multi-channel EEG-like data.

    Class 1 gets an extra ~10 Hz oscillation concentrated in the first
    half of channels; class 2 gets it in the second half -- gives CSP
    genuine, findable class-discriminative spatial structure, loosely
    analogous to real ERD/ERS lateralisation without being derived from
    any real recording.
    """
    rng = np.random.default_rng(seed)
    n_trials = 2 * n_trials_per_class
    X = np.zeros((n_trials, n_channels, n_samples))
    y = np.zeros(n_trials, dtype=int)
    t = np.arange(n_samples) / sfreq
    half = n_channels // 2

    for trial in range(n_trials):
        cls = 1 if trial < n_trials_per_class else 2
        y[trial] = cls
        for ch in range(n_channels):
            # 1/f-ish background: sum of a few low-frequency components
            # with decaying amplitude, plus white noise -- looks more
            # EEG-like than pure white noise.
            background = np.zeros(n_samples)
            for k, f0 in enumerate([2, 5, 11, 19, 27, 35], start=1):
                background += (rng.normal(0.5, 0.15) / k) * np.sin(
                    2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
            background += rng.normal(0, 0.3, n_samples)

            # Class-discriminative mu-band (~10 Hz) burst, concentrated
            # in different channels per class -- envelope tapers at the
            # trial edges so it doesn't look like an abrupt on/off switch.
            is_discriminative_channel = (ch < half) if cls == 1 else (ch >= half)
            if is_discriminative_channel:
                envelope = np.sin(np.pi * t / t[-1]) ** 2  # 0 -> 1 -> 0
                burst = 1.8 * envelope * np.sin(2 * np.pi * 10.5 * t +
                                                  rng.uniform(0, 2 * np.pi))
                background += burst

            X[trial, ch, :] = background

    return X, y


def apply_filter_bank(X: np.ndarray, bands, sfreq: float,
                       order: int = 4) -> list:
    """Causal Butterworth filter bank -- matches preprocessing.py's real
    bandpass_filter()/apply_filter_bank() exactly (sosfilt, not filtfilt,
    so it's a single forward pass -- the same causal design used for the
    real analog-circuit-mapping argument)."""
    nyq = sfreq / 2.0
    filtered = []
    for lo, hi in bands:
        sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
        filtered.append(sosfilt(sos, X, axis=-1))
    return filtered


def regularise(cov: np.ndarray, lambda_r: float) -> np.ndarray:
    """Tikhonov regularisation: Sigma_reg = (1 - lambda) Sigma + lambda I."""
    n = cov.shape[0]
    return (1 - lambda_r) * cov + lambda_r * np.eye(n)


def fit_csp(X_band: np.ndarray, y: np.ndarray, m: int,
            lambda_r: float) -> np.ndarray:
    """Dual-end CSP for one band, one class pair (binary here).

    Solves the generalised eigenvalue problem Sigma_A W = lambda (Sigma_A
    + Sigma_B) W via scipy.linalg.eigh, matching preprocessing.py's
    _solve_csp exactly -- takes m eigenvectors from EACH end of the
    spectrum (2m total), not just the largest-eigenvalue end.
    """
    from scipy.linalg import eigh

    classes = np.unique(y)
    covs = {}
    for cls in classes:
        Xc = X_band[y == cls]  # (n_trials_c, n_channels, n_samples)
        # Per-trial normalised covariance, averaged across trials
        # (arithmetic mean here for simplicity -- real code defaults to
        # a Riemannian mean, but arithmetic is a reasonable stand-in for
        # a synthetic illustration with well-separated classes).
        trial_covs = [np.cov(Xc[t_i]) / np.trace(np.cov(Xc[t_i]))
                      for t_i in range(Xc.shape[0])]
        covs[cls] = regularise(np.mean(trial_covs, axis=0), lambda_r)

    cov_a, cov_b = covs[classes[0]], covs[classes[1]]
    eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
    # Dual-end: m smallest + m largest eigenvalues' eigenvectors.
    W = np.concatenate([eigvecs[:, :m], eigvecs[:, -m:]], axis=1)
    return W  # (n_channels, 2m)


def znorm_fit_transform(X: np.ndarray) -> np.ndarray:
    """Per-feature z-normalisation, matching preprocessing.py's ZNormaliser."""
    n_trials, n_features, n_samples = X.shape
    X_2d = X.transpose(1, 0, 2).reshape(n_features, n_trials * n_samples)
    mean = X_2d.mean(axis=1)
    std = X_2d.std(axis=1) + 1e-8
    return (X - mean[None, :, None]) / std[None, :, None]


def encode_adaptive_threshold(X: np.ndarray, base_thresh: float,
                               adapt_inc: float, decay: float) -> np.ndarray:
    """Adaptive-threshold delta encoder, matching encoding.py's real
    corrected update rule: theta(t+1) = decay*theta(t) + adapt_inc*s(t)
    (decay always applies; increment is added on top when a spike fires).

    Parameters
    ----------
    X : np.ndarray
        Shape (n_trials, n_features, n_samples).
    """
    n_trials, n_features, n_samples = X.shape
    spikes = np.zeros_like(X)
    threshold = np.full((n_trials, n_features), base_thresh)
    for t_i in range(1, n_samples):
        delta = np.abs(X[:, :, t_i] - X[:, :, t_i - 1])
        fired = (delta > threshold).astype(float)
        spikes[:, :, t_i] = fired
        threshold = threshold * decay + fired * adapt_inc
    return spikes


def _stacked_plot(ax, time_s, traces, labels):
    n = len(traces)
    scale = max(np.max(np.abs(t)) for t in traces) if traces else 1.0
    offset_step = scale * 2.2
    for i, (trace, label) in enumerate(zip(traces, labels)):
        offset = (n - 1 - i) * offset_step
        ax.plot(time_s, trace + offset, linewidth=0.7)
        ax.text(-0.02, offset, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8)
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--duration-s", type=float, default=4.0)
    ap.add_argument("--n-channels", type=int, default=8)
    ap.add_argument("--n-trials-per-class", type=int, default=30)
    ap.add_argument("--trial-idx", type=int, default=0,
                     help="Which trial (within its class) to visualise.")
    ap.add_argument("--band-idx", type=int, default=1,
                     help="Which of the 6 fixed bands to show as "
                          "'filtered' (default 1 = 8-14 Hz).")
    ap.add_argument("--n-lines", type=int, default=6,
                     help="Number of channels/CSP-features to show, stacked.")
    ap.add_argument("--m", type=int, default=4,
                     help="CSP eigenvectors per end (matches real default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="figures/pipeline_signals_synthetic.png")
    args = ap.parse_args()

    n_samples = int(args.duration_s * args.sfreq)

    print("Generating synthetic 2-class EEG-like signals ...")
    X, y = make_synthetic_trials(args.n_trials_per_class, args.n_channels,
                                  n_samples, args.sfreq, args.seed)
    print(f"  X: {X.shape}  (n_trials, n_channels, n_samples), "
          f"classes: {np.unique(y)}")

    trial_idx = args.trial_idx  # index into class-1's trials
    n_lines = min(args.n_lines, args.n_channels)
    ch_idx = list(range(n_lines))
    ch_labels = [f"ch{i}" for i in ch_idx]

    # ---- Stage 1: raw ----
    raw_traces = [X[trial_idx, c, :] for c in ch_idx]

    # ---- Stage 2: fixed six-band filter bank ----
    print("Applying fixed six-band filter bank ...")
    X_bands = apply_filter_bank(X, FREQ_BANDS, args.sfreq)
    filtered_traces = [X_bands[args.band_idx][trial_idx, c, :] for c in ch_idx]

    # ---- Stage 3: dual-end CSP (fit across ALL bands, concatenate) ----
    print("Fitting dual-end CSP across all 6 bands ...")
    proj_bands = []
    for X_band in X_bands:
        W = fit_csp(X_band, y, args.m, LAMBDA_R)          # (n_channels, 2m)
        proj = np.einsum("cf,tcs->tfs", W, X_band)         # (n_trials, 2m, n_samples)
        proj_bands.append(proj)
    X_concat = np.concatenate(proj_bands, axis=1)  # (n_trials, 6*2m, n_samples)
    print(f"  CSP-projected: {X_concat.shape}")
    n_feat = min(args.n_lines, X_concat.shape[1])
    feat_idx = list(range(n_feat))
    feat_labels = [f"CSP feat. {i}" for i in feat_idx]
    csp_traces = [X_concat[trial_idx, f, :] for f in feat_idx]

    # ---- Stage 4: z-normalisation ----
    X_norm = znorm_fit_transform(X_concat)
    znorm_traces = [X_norm[trial_idx, f, :] for f in feat_idx]

    # ---- Stage 5: adaptive-threshold spike encoding ----
    print("Encoding spikes (adaptive-threshold delta encoder) ...")
    spikes = encode_adaptive_threshold(X_norm, BASE_THRESH, ADAPT_INC, DECAY)
    spike_traces = [spikes[trial_idx, f, :] for f in feat_idx]

    # ---- Plot ----
    time_s = np.arange(n_samples) / args.sfreq
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)

    _stacked_plot(axes[0], time_s, raw_traces, ch_labels)
    axes[0].set_title(f"1. Raw (synthetic) EEG ({n_lines} channels)")

    lo, hi = FREQ_BANDS[args.band_idx]
    _stacked_plot(axes[1], time_s, filtered_traces, ch_labels)
    axes[1].set_title(f"2. After causal bandpass filter ({lo}-{hi} Hz band)")

    _stacked_plot(axes[2], time_s, csp_traces, feat_labels)
    axes[2].set_title(f"3. After dual-end pairwise CSP "
                       f"({n_feat} spatial-filter outputs)")

    _stacked_plot(axes[3], time_s, znorm_traces, feat_labels)
    axes[3].set_title("4. After z-normalisation")

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

    fig.suptitle(f"SYNTHETIC illustration (class {int(y[trial_idx])}): "
                 f"signal through the pipeline -- for explanation only, "
                 f"not real data", fontsize=11)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"Saved: {args.output} (and matching .pdf)")


if __name__ == "__main__":
    main()
