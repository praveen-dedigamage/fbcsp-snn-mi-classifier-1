"""Cross-subject accuracy analysis.

Reads ``Results/Subject_N/summary.csv`` for each requested subject, computes
per-subject FP32 and INT8 mean ± std over CV folds, prints a formatted table,
and saves a grouped bar chart.

Usage
-----
    python analyze_results.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9

    # Custom results directory or a subset of subjects
    python analyze_results.py --results-dir /scratch/my_run/Results --subjects 1 3 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _load_summary(csv_path: Path) -> list[dict]:
    """Load a summary.csv and return a list of per-fold dicts.

    Skips the 'mean' row written by run_aggregate so that the statistics
    computed here are consistent with the raw per-fold values.

    Parameters
    ----------
    csv_path : Path
        Path to ``summary.csv``.

    Returns
    -------
    list[dict]
        One dict per numeric fold row.  Keys: ``fold``, ``test_acc_fp32``,
        ``test_acc_int8``, ``best_val_acc_fp32``.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("fold", "").strip().lower() in ("", "mean"):
                continue
            entry: dict = {
                "fold":              int(row["fold"]),
                "test_acc_fp32":     float(row["test_acc_fp32"]),
                "test_acc_int8":     float(row["test_acc_int8"]),
                "best_val_acc_fp32": float(row["best_val_acc_fp32"]),
            }
            # Baseline columns are optional (absent in runs before baseline.py was added)
            if row.get("test_acc_lda", "").strip():
                entry["test_acc_lda"] = float(row["test_acc_lda"])
            if row.get("test_acc_svm", "").strip():
                entry["test_acc_svm"] = float(row["test_acc_svm"])
            rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Per-subject statistics
# ---------------------------------------------------------------------------

def _subject_stats(rows: list[dict]) -> dict:
    """Compute mean and std for each metric over fold rows.

    Parameters
    ----------
    rows : list[dict]
        Output of :func:`_load_summary`.

    Returns
    -------
    dict
        Keys: ``fp32_mean``, ``fp32_std``, ``int8_mean``, ``int8_std``,
        ``val_mean``, ``val_std``, ``n_folds``.  All accuracies in [0, 1].
    """
    fp32 = np.array([r["test_acc_fp32"]     for r in rows])
    int8 = np.array([r["test_acc_int8"]     for r in rows])
    val  = np.array([r["best_val_acc_fp32"] for r in rows])
    result: dict = {
        "fp32_mean": float(fp32.mean()),
        "fp32_std":  float(fp32.std()),
        "int8_mean": float(int8.mean()),
        "int8_std":  float(int8.std()),
        "val_mean":  float(val.mean()),
        "val_std":   float(val.std()),
        "n_folds":   len(rows),
    }
    lda_vals = [r["test_acc_lda"] for r in rows if "test_acc_lda" in r]
    svm_vals = [r["test_acc_svm"] for r in rows if "test_acc_svm" in r]
    if lda_vals:
        lda = np.array(lda_vals)
        result["lda_mean"] = float(lda.mean())
        result["lda_std"]  = float(lda.std())
    if svm_vals:
        svm = np.array(svm_vals)
        result["svm_mean"] = float(svm.mean())
        result["svm_std"]  = float(svm.std())
    return result


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

_COL_WIDTH = 16

def _print_table(
    subjects: list[int],
    stats: dict[int, dict],
    missing: list[int],
) -> None:
    """Print a formatted cross-subject accuracy table to stdout."""

    # Detect whether baseline columns are present in any subject
    has_lda = any("lda_mean" in stats.get(s, {}) for s in subjects)
    has_svm = any("svm_mean" in stats.get(s, {}) for s in subjects)

    if has_lda or has_svm:
        header_fmt = "{:<8} {:>14} {:>12} {:>12} {:>8}"
        row_fmt    = "{:<8} {:>14} {:>12} {:>12} {:>8}"
        sep = "-" * 58
        print()
        print("Cross-Subject Accuracy — FBCSP-SNN vs Classical Baselines (BNCI2014_001)")
        print(sep)
        print(header_fmt.format("Subject", "SNN FP32 (%)", "LDA (%)", "SVM (%)", "Folds"))
        print(sep)
    else:
        header_fmt = "{:<8} {:>14} {:>14} {:>14} {:>8}"
        row_fmt    = "{:<8} {:>14} {:>14} {:>14} {:>8}"
        sep = "-" * 62
        print()
        print("Cross-Subject Accuracy Summary — FBCSP-SNN (BNCI2014_001)")
        print(sep)
        print(header_fmt.format("Subject", "Test FP32 (%)", "Test INT8 (%)", "Val FP32 (%)", "Folds"))
        print(sep)

    fp32_vals, int8_vals, lda_vals, svm_vals = [], [], [], []

    for s in subjects:
        if s in missing:
            print(row_fmt.format(f"S{s}", "--", "--", "--", "--"))
            continue
        st = stats[s]
        fp32_str = f"{st['fp32_mean']*100:.1f} ± {st['fp32_std']*100:.1f}"

        if has_lda or has_svm:
            lda_str = f"{st['lda_mean']*100:.1f} ± {st['lda_std']*100:.1f}" if "lda_mean" in st else "  N/A"
            svm_str = f"{st['svm_mean']*100:.1f} ± {st['svm_std']*100:.1f}" if "svm_mean" in st else "  N/A"
            print(row_fmt.format(f"S{s}", fp32_str, lda_str, svm_str, st["n_folds"]))
            if "lda_mean" in st: lda_vals.append(st["lda_mean"])
            if "svm_mean" in st: svm_vals.append(st["svm_mean"])
        else:
            int8_str = f"{st['int8_mean']*100:.1f} ± {st['int8_std']*100:.1f}"
            val_str  = f"{st['val_mean']*100:.1f} ± {st['val_std']*100:.1f}"
            print(row_fmt.format(f"S{s}", fp32_str, int8_str, val_str, st["n_folds"]))
            int8_vals.append(st["int8_mean"])

        fp32_vals.append(st["fp32_mean"])

    print(sep)

    if fp32_vals:
        grand_fp32     = np.mean(fp32_vals)
        grand_fp32_std = np.std(fp32_vals)
        total_folds    = sum(stats[s]["n_folds"] for s in subjects if s not in missing)

        if has_lda or has_svm:
            grand_lda = f"{np.mean(lda_vals)*100:.1f}" if lda_vals else "N/A"
            grand_svm = f"{np.mean(svm_vals)*100:.1f}" if svm_vals else "N/A"
            print(row_fmt.format(
                "GRAND",
                f"{grand_fp32*100:.1f} ± {grand_fp32_std*100:.1f}",
                grand_lda, grand_svm, total_folds,
            ))
        else:
            grand_int8     = np.mean(int8_vals)
            grand_int8_std = np.std(int8_vals)
            print(row_fmt.format(
                "GRAND",
                f"{grand_fp32*100:.1f} ± {grand_fp32_std*100:.1f}",
                f"{grand_int8*100:.1f} ± {grand_int8_std*100:.1f}",
                "", total_folds,
            ))

        print(sep)
        print(f"  Baseline target : 64.8% (FP32, static bands, 22 CSP comps)")
        print(f"  Target          : 70.0%+")
        delta = grand_fp32 * 100 - 64.8
        sign  = "+" if delta >= 0 else ""
        print(f"  SNN vs baseline : {sign}{delta:.1f} pp")

        if lda_vals:
            delta_lda = np.mean(lda_vals) * 100 - 64.8
            sign_lda  = "+" if delta_lda >= 0 else ""
            print(f"  LDA vs baseline : {sign_lda}{delta_lda:.1f} pp")
        if svm_vals:
            delta_svm = np.mean(svm_vals) * 100 - 64.8
            sign_svm  = "+" if delta_svm >= 0 else ""
            print(f"  SVM vs baseline : {sign_svm}{delta_svm:.1f} pp")

    print()
    if missing:
        print(f"  WARNING: Results missing for subjects: {missing}")
        print()


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def _plot_bar_chart(
    subjects: list[int],
    stats: dict[int, dict],
    save_path: Path,
    baseline_fp32: float = 0.648,
) -> None:
    """Save a grouped bar chart comparing per-subject FP32 and INT8 accuracy.

    Parameters
    ----------
    subjects : list[int]
        Subject IDs to include (in display order).
    stats : dict[int, dict]
        Output of :func:`_subject_stats` per subject.
    save_path : Path
        Destination PNG.  Parent directory is created if absent.
    baseline_fp32 : float
        Dashed reference line for the previous-pipeline baseline.
    """
    valid = [s for s in subjects if s in stats]
    if not valid:
        print("No data to plot.")
        return

    has_lda = any("lda_mean" in stats[s] for s in valid)
    has_svm = any("svm_mean" in stats[s] for s in valid)

    x = np.arange(len(valid))

    fp32_means = np.array([stats[s]["fp32_mean"] * 100 for s in valid])
    fp32_stds  = np.array([stats[s]["fp32_std"]  * 100 for s in valid])

    if has_lda or has_svm:
        # 3 or 4 bars per subject: SNN / LDA / SVM
        n_bars = 1 + int(has_lda) + int(has_svm)
        width  = 0.8 / n_bars
        offsets: list[float] = []
        labels_bars: list[str] = ["SNN (FP32)"]
        colors: list[str] = ["steelblue"]
        if has_lda:
            offsets.append(-width if n_bars == 3 else -width * 1.5)
            labels_bars.append("LDA")
            colors.append("seagreen")
        if has_svm:
            offsets.append(width if n_bars == 3 else -width * 0.5)
            labels_bars.append("SVM")
            colors.append("darkorange")
        snn_offset = 0.0 if n_bars == 2 else (width if has_lda and has_svm else 0.0)

        fig, ax = plt.subplots(figsize=(max(10, len(valid) * 1.5), 5))

        bars_snn = ax.bar(
            x + snn_offset, fp32_means, width,
            yerr=fp32_stds, capsize=3,
            color="steelblue", alpha=0.85, label="SNN (FP32)",
            error_kw={"elinewidth": 1.0, "ecolor": "navy"},
        )
        for bar, val in zip(bars_snn, fp32_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, color="navy")

        bar_idx = 0
        if has_lda:
            lda_means = np.array([stats[s].get("lda_mean", 0) * 100 for s in valid])
            lda_stds  = np.array([stats[s].get("lda_std",  0) * 100 for s in valid])
            off = offsets[bar_idx]; bar_idx += 1
            bars_lda = ax.bar(
                x + off, lda_means, width,
                yerr=lda_stds, capsize=3,
                color="seagreen", alpha=0.80, label="LDA",
                error_kw={"elinewidth": 1.0, "ecolor": "darkgreen"},
            )
            for bar, val in zip(bars_lda, lda_means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, color="darkgreen")

        if has_svm:
            svm_means = np.array([stats[s].get("svm_mean", 0) * 100 for s in valid])
            svm_stds  = np.array([stats[s].get("svm_std",  0) * 100 for s in valid])
            off = offsets[bar_idx]
            bars_svm = ax.bar(
                x + off, svm_means, width,
                yerr=svm_stds, capsize=3,
                color="darkorange", alpha=0.80, label="SVM",
                error_kw={"elinewidth": 1.0, "ecolor": "saddlebrown"},
            )
            for bar, val in zip(bars_svm, svm_means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, color="saddlebrown")
    else:
        # Original 2-bar layout: FP32 + INT8
        width = 0.35
        int8_means = np.array([stats[s]["int8_mean"] * 100 for s in valid])
        int8_stds  = np.array([stats[s]["int8_std"]  * 100 for s in valid])

        fig, ax = plt.subplots(figsize=(max(9, len(valid) * 1.2), 5))

        bars_fp32 = ax.bar(
            x - width / 2, fp32_means, width,
            yerr=fp32_stds, capsize=4,
            color="steelblue", alpha=0.85, label="FP32",
            error_kw={"elinewidth": 1.2, "ecolor": "navy"},
        )
        bars_int8 = ax.bar(
            x + width / 2, int8_means, width,
            yerr=int8_stds, capsize=4,
            color="darkorange", alpha=0.80, label="INT8 (sim)",
            error_kw={"elinewidth": 1.2, "ecolor": "saddlebrown"},
        )
        for bar, val in zip(bars_fp32, fp32_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="navy")
        for bar, val in zip(bars_int8, int8_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="saddlebrown")

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in valid])
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "FBCSP-SNN Cross-Subject Accuracy — BNCI2014_001\n"
        "(mean ± std over CV folds)",
        fontsize=13,
    )
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-subject accuracy analysis for the FBCSP-SNN pipeline."
    )
    p.add_argument(
        "--results-dir", default="Results",
        help="Root results directory (default: Results)",
    )
    p.add_argument(
        "--subjects", nargs="+", type=int,
        default=list(range(1, 10)),
        help="Subject IDs to include (default: 1 2 3 4 5 6 7 8 9)",
    )
    p.add_argument(
        "--output", default=None,
        help="Path for the bar chart PNG.  "
             "Defaults to <results-dir>/cross_subject_accuracy.png",
    )
    p.add_argument(
        "--baseline", type=float, default=0.648,
        help="Previous-pipeline baseline FP32 accuracy to show as reference "
             "(default: 0.648 = 64.8%%)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)

    stats:   dict[int, dict] = {}
    missing: list[int]       = []

    for s in args.subjects:
        csv_path = results_dir / f"Subject_{s}" / "summary.csv"
        if not csv_path.exists():
            missing.append(s)
            continue
        rows = _load_summary(csv_path)
        if not rows:
            print(
                f"  WARNING: Subject {s} summary.csv exists but has no fold rows "
                f"(run_aggregate may not have been called yet).",
                file=sys.stderr,
            )
            missing.append(s)
            continue
        stats[s] = _subject_stats(rows)

    if not stats:
        print(
            f"No summary.csv files found under {results_dir}.\n"
            "Run training and aggregation first:\n"
            "  python main.py train   --subject-id N --n-folds 10\n"
            "  python main.py aggregate --subject-id N --n-folds 10",
            file=sys.stderr,
        )
        sys.exit(1)

    _print_table(args.subjects, stats, missing)

    chart_path = (
        Path(args.output)
        if args.output
        else results_dir / "cross_subject_accuracy.png"
    )
    _plot_bar_chart(args.subjects, stats, chart_path, baseline_fp32=args.baseline)


if __name__ == "__main__":
    main()
