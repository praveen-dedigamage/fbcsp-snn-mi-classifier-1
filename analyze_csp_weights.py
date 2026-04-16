"""CSP weight matrix analysis — distribution, quantization simulation, SQNR.

Loads all csp_filters.pkl files from a results directory, extracts spatial
filter matrices, and produces:

  1. Global value histogram with per-bit-width step-size markers
  2. Per-subject SQNR vs bit-width curve
  3. Per-subject quantization RMSE vs bit-width
  4. Summary statistics table (range, std, kurtosis, percentiles)

No GPU or PyTorch required — pure NumPy / matplotlib.

Usage
-----
    # On Puhti login node (activate venv first):
    python analyze_csp_weights.py --results-dir Results_adm_static6 \
                                  --out-dir Results_adm_static6/csp_analysis \
                                  --bits 4 6 8

    # Quick single-subject check:
    python analyze_csp_weights.py --results-dir Results_adm_static6 \
                                  --subjects 1 2 3 --bits 4 6 8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def symmetric_quantize(W: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric per-tensor quantization of a weight matrix.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix, any shape.
    bits : int
        Bit-width (e.g. 4, 6, 8).

    Returns
    -------
    np.ndarray
        Quantized weight matrix, same shape as W.
    """
    max_val = np.max(np.abs(W))
    if max_val < 1e-12:
        return W.copy()
    n_levels = 2 ** (bits - 1) - 1          # e.g. 7 for 4-bit, 31 for 6-bit
    scale = max_val / n_levels
    W_q = np.clip(np.round(W / scale), -n_levels, n_levels) * scale
    return W_q


def sqnr(W: np.ndarray, W_q: np.ndarray) -> float:
    """Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = np.sum(W ** 2)
    noise_power  = np.sum((W - W_q) ** 2)
    if noise_power < 1e-20:
        return np.inf
    return float(10 * np.log10(signal_power / noise_power))


def norm_rmse(W: np.ndarray, W_q: np.ndarray) -> float:
    """Normalised RMSE: RMS(error) / RMS(signal)."""
    signal_rms = np.sqrt(np.mean(W ** 2))
    if signal_rms < 1e-12:
        return 0.0
    return float(np.sqrt(np.mean((W - W_q) ** 2)) / signal_rms)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csp_weights(
    results_dir: Path,
    subjects: List[int],
) -> Dict[int, List[np.ndarray]]:
    """Load all CSP filter matrices from csp_filters.pkl files.

    Parameters
    ----------
    results_dir : Path
        Root results directory containing Subject_N/fold_K/csp_filters.pkl.
    subjects : List[int]
        Subject IDs to load.

    Returns
    -------
    Dict[int, List[np.ndarray]]
        Maps subject_id → flat list of W matrices (one per band×pair×fold).
    """
    subject_weights: Dict[int, List[np.ndarray]] = {}

    for s in subjects:
        subj_dir = results_dir / f"Subject_{s}"
        if not subj_dir.exists():
            print(f"  [WARN] Subject_{s} not found in {results_dir} — skip")
            continue

        all_W: List[np.ndarray] = []
        fold_dirs = sorted(subj_dir.glob("fold_*"))
        if not fold_dirs:
            print(f"  [WARN] No fold directories for Subject_{s} — skip")
            continue

        for fold_dir in fold_dirs:
            pkl_path = fold_dir / "csp_filters.pkl"
            if not pkl_path.exists():
                print(f"  [WARN] Missing {pkl_path} — skip")
                continue
            with open(pkl_path, "rb") as f:
                csp = pickle.load(f)
            for W in csp.filters_.values():
                all_W.append(W.astype(np.float64))

        if all_W:
            subject_weights[s] = all_W
            n_vals = sum(w.size for w in all_W)
            print(f"  S{s}: {len(all_W)} matrices, "
                  f"{n_vals:,} weight values "
                  f"(shape example: {all_W[0].shape})")

    return subject_weights


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    weights: List[np.ndarray],
    bits_list: List[int],
) -> dict:
    """Compute distribution and quantization statistics for a list of matrices."""
    flat = np.concatenate([w.ravel() for w in weights])

    stats: dict = {
        "n_values":   len(flat),
        "min":        float(flat.min()),
        "max":        float(flat.max()),
        "abs_max":    float(np.abs(flat).max()),
        "mean":       float(flat.mean()),
        "std":        float(flat.std()),
        "p1":         float(np.percentile(flat, 1)),
        "p5":         float(np.percentile(flat, 5)),
        "p95":        float(np.percentile(flat, 95)),
        "p99":        float(np.percentile(flat, 99)),
        "kurtosis":   float(kurtosis(flat, fisher=True)),   # excess kurtosis
    }

    # Step size and SQNR per bit-width (per-tensor over all values together)
    for bits in bits_list:
        n_levels = 2 ** (bits - 1) - 1
        scale    = stats["abs_max"] / n_levels
        W_all    = flat
        W_q_all  = np.clip(np.round(W_all / scale), -n_levels, n_levels) * scale
        stats[f"step_{bits}b"]      = float(scale)
        stats[f"sqnr_{bits}b"]      = sqnr(W_all, W_q_all)
        stats[f"nrmse_{bits}b"]     = norm_rmse(W_all, W_q_all)
        stats[f"outlier_frac_{bits}b"] = float(
            np.mean(np.abs(W_all) > n_levels * scale * 0.95)
        )

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histogram(
    all_flat: np.ndarray,
    bits_list: List[int],
    out_path: Path,
) -> None:
    """Global histogram with per-bit step-size markers."""
    abs_max = float(np.abs(all_flat).max())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(all_flat, bins=200, color="steelblue", alpha=0.75, density=True,
            label="CSP weight values (all subjects/folds/bands/pairs)")

    colors = {4: "#e74c3c", 6: "#e67e22", 8: "#27ae60"}
    linestyles = {4: "--", 6: "-.", 8: ":"}

    for bits in bits_list:
        n_levels = 2 ** (bits - 1) - 1
        step = abs_max / n_levels
        for k in range(-n_levels, n_levels + 1):
            xval = k * step
            ax.axvline(xval, color=colors[bits], alpha=0.35,
                       linewidth=0.7, linestyle=linestyles[bits])
        # Legend proxy
        ax.axvline(np.nan, color=colors[bits], linewidth=1.5,
                   linestyle=linestyles[bits],
                   label=f"{bits}-bit  (step={step:.4f},  {2**bits} levels)")

    ax.set_xlabel("CSP eigenvector value")
    ax.set_ylabel("Density")
    ax.set_title("CSP Spatial Filter Weight Distribution\nwith quantization grid overlaid")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sqnr_curves(
    subject_stats: Dict[int, dict],
    bits_list: List[int],
    out_path: Path,
) -> None:
    """Per-subject SQNR vs bit-width."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for s, stats in sorted(subject_stats.items()):
        sqnr_vals = [stats[f"sqnr_{b}b"] for b in bits_list]
        ax.plot(bits_list, sqnr_vals, marker="o", label=f"S{s}")

    ax.set_xlabel("Bit-width")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("CSP Weight SQNR vs Quantization Bit-width")
    ax.set_xticks(bits_list)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_nrmse_curves(
    subject_stats: Dict[int, dict],
    bits_list: List[int],
    out_path: Path,
) -> None:
    """Per-subject normalised RMSE vs bit-width."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for s, stats in sorted(subject_stats.items()):
        nrmse_vals = [stats[f"nrmse_{b}b"] * 100 for b in bits_list]
        ax.plot(bits_list, nrmse_vals, marker="o", label=f"S{s}")

    ax.axhline(5.0, color="red", linestyle="--", linewidth=1,
               label="5% threshold")
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("Normalised RMSE (%)")
    ax.set_title("CSP Weight Normalised RMSE vs Bit-width")
    ax.set_xticks(bits_list)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_table(
    subject_stats: Dict[int, dict],
    bits_list: List[int],
) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 90)
    print("  CSP WEIGHT ANALYSIS SUMMARY")
    print("=" * 90)

    # Distribution stats
    print(f"\n{'Subj':>5}  {'N vals':>8}  {'Min':>8}  {'Max':>8}  "
          f"{'Std':>8}  {'p1':>8}  {'p99':>8}  {'Kurt':>7}")
    print("-" * 75)
    for s, st in sorted(subject_stats.items()):
        print(f"  S{s:>2}  {st['n_values']:>8,}  {st['min']:>8.4f}  "
              f"{st['max']:>8.4f}  {st['std']:>8.4f}  "
              f"{st['p1']:>8.4f}  {st['p99']:>8.4f}  "
              f"{st['kurtosis']:>7.2f}")

    # Quantization stats
    for bits in bits_list:
        print(f"\n  {bits}-bit quantization  "
              f"(levels: ±{2**(bits-1)-1}, step = abs_max / {2**(bits-1)-1})")
        print(f"  {'Subj':>5}  {'Step':>9}  {'SQNR(dB)':>10}  "
              f"{'NRMSE(%)':>10}  {'Outlier%':>10}")
        print("  " + "-" * 50)
        for s, st in sorted(subject_stats.items()):
            print(f"  S{s:>2}  "
                  f"{st[f'step_{bits}b']:>9.5f}  "
                  f"{st[f'sqnr_{bits}b']:>10.2f}  "
                  f"{st[f'nrmse_{bits}b']*100:>10.3f}  "
                  f"{st[f'outlier_frac_{bits}b']*100:>10.2f}")
    print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse CSP weight distributions and simulate quantization."
    )
    parser.add_argument("--results-dir", type=str, default="Results_adm_static6",
                        help="Root results directory (default: Results_adm_static6)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for plots (default: <results-dir>/csp_analysis)")
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)),
                        help="Subject IDs to analyse (default: 1-9)")
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8],
                        help="Bit-widths to simulate (default: 4 6 8)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "csp_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults dir : {results_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Subjects    : {args.subjects}")
    print(f"Bit-widths  : {args.bits}")
    print()

    # Load
    print("Loading CSP pickle files...")
    subject_weights = load_csp_weights(results_dir, args.subjects)
    if not subject_weights:
        print("ERROR: no CSP weights found. Check --results-dir path.")
        sys.exit(1)

    # Per-subject stats
    print("\nComputing statistics...")
    subject_stats: Dict[int, dict] = {}
    for s, weights in subject_weights.items():
        subject_stats[s] = compute_stats(weights, args.bits)

    # Global flat array for histogram
    all_flat = np.concatenate([
        w.ravel()
        for weights in subject_weights.values()
        for w in weights
    ])

    print_summary_table(subject_stats, args.bits)

    # Plots
    print("Generating plots...")
    plot_histogram(all_flat, args.bits,
                   out_dir / "csp_weight_histogram.png")
    plot_sqnr_curves(subject_stats, args.bits,
                     out_dir / "csp_sqnr_vs_bits.png")
    plot_nrmse_curves(subject_stats, args.bits,
                      out_dir / "csp_nrmse_vs_bits.png")

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
