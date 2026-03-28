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
            rows.append({
                "fold":             int(row["fold"]),
                "test_acc_fp32":    float(row["test_acc_fp32"]),
                "test_acc_int8":    float(row["test_acc_int8"]),
                "best_val_acc_fp32": float(row["best_val_acc_fp32"]),
            })
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
    fp32 = np.array([r["test_acc_fp32"]    for r in rows])
    int8 = np.array([r["test_acc_int8"]    for r in rows])
    val  = np.array([r["best_val_acc_fp32"] for r in rows])
    return {
        "fp32_mean": float(fp32.mean()),
        "fp32_std":  float(fp32.std()),
        "int8_mean": float(int8.mean()),
        "int8_std":  float(int8.std()),
        "val_mean":  float(val.mean()),
        "val_std":   float(val.std()),
        "n_folds":   len(rows),
    }


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

    header_fmt = "{:<8} {:>14} {:>14} {:>14} {:>8}"
    row_fmt    = "{:<8} {:>14} {:>14} {:>14} {:>8}"
    sep = "-" * 62

    print()
    print("Cross-Subject Accuracy Summary — FBCSP-SNN (BNCI2014_001)")
    print(sep)
    print(header_fmt.format(
        "Subject", "Test FP32 (%)", "Test INT8 (%)", "Val FP32 (%)", "Folds"
    ))
    print(sep)

    fp32_vals, int8_vals = [], []

    for s in subjects:
        if s in missing:
            print(row_fmt.format(f"S{s}", "--", "--", "--", "--"))
            continue
        st = stats[s]
        fp32_str = f"{st['fp32_mean']*100:.1f} ± {st['fp32_std']*100:.1f}"
        int8_str = f"{st['int8_mean']*100:.1f} ± {st['int8_std']*100:.1f}"
        val_str  = f"{st['val_mean']*100:.1f} ± {st['val_std']*100:.1f}"
        print(row_fmt.format(f"S{s}", fp32_str, int8_str, val_str, st["n_folds"]))
        fp32_vals.append(st["fp32_mean"])
        int8_vals.append(st["int8_mean"])

    print(sep)

    if fp32_vals:
        grand_fp32 = np.mean(fp32_vals)
        grand_int8 = np.mean(int8_vals)
        grand_fp32_std = np.std(fp32_vals)
        grand_int8_std = np.std(int8_vals)
        print(row_fmt.format(
            "GRAND",
            f"{grand_fp32*100:.1f} ± {grand_fp32_std*100:.1f}",
            f"{grand_int8*100:.1f} ± {grand_int8_std*100:.1f}",
            "",
            sum(stats[s]["n_folds"] for s in subjects if s not in missing),
        ))
        print(sep)
        print(f"  Baseline target : 64.8% (FP32, static bands, 22 CSP comps)")
        print(f"  Target          : 70.0%+")
        delta = grand_fp32 * 100 - 64.8
        sign  = "+" if delta >= 0 else ""
        print(f"  vs. baseline    : {sign}{delta:.1f} pp")

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

    x = np.arange(len(valid))
    width = 0.35

    fp32_means = np.array([stats[s]["fp32_mean"] * 100 for s in valid])
    fp32_stds  = np.array([stats[s]["fp32_std"]  * 100 for s in valid])
    int8_means = np.array([stats[s]["int8_mean"] * 100 for s in valid])
    int8_stds  = np.array([stats[s]["int8_std"]  * 100 for s in valid])

    fig, ax = plt.subplots(figsize=(max(9, len(valid) * 1.2), 5))

    bars_fp32 = ax.bar(
        x - width / 2, fp32_means, width,
        yerr=fp32_stds, capsize=4,
        color="steelblue", alpha=0.85,
        label="FP32",
        error_kw={"elinewidth": 1.2, "ecolor": "navy"},
    )
    bars_int8 = ax.bar(
        x + width / 2, int8_means, width,
        yerr=int8_stds, capsize=4,
        color="darkorange", alpha=0.80,
        label="INT8 (sim)",
        error_kw={"elinewidth": 1.2, "ecolor": "saddlebrown"},
    )

    # Baseline reference
    ax.axhline(
        baseline_fp32 * 100, color="crimson", lw=1.5, ls="--",
        label=f"Previous baseline ({baseline_fp32*100:.1f}%)",
    )
    # Target line
    ax.axhline(
        70.0, color="green", lw=1.2, ls=":",
        label="Target (70%)",
    )
    # Chance level (25% for 4-class)
    ax.axhline(
        25.0, color="grey", lw=0.8, ls=":",
        label="Chance (25%)",
    )

    # Annotate bar tops
    for bar, val in zip(bars_fp32, fp32_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="navy",
        )
    for bar, val in zip(bars_int8, int8_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="saddlebrown",
        )

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
