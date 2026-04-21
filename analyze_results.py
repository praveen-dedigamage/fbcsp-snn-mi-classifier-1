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
            # PTQ-CSP columns (present in runs that integrated per-fold PTQ sweep)
            for col in ("test_acc_csp_8bit", "test_acc_csp_6bit", "test_acc_csp_4bit"):
                if row.get(col, "").strip():
                    entry[col] = float(row[col])
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
    for col, key in (
        ("test_acc_csp_8bit", "csp8_mean"),
        ("test_acc_csp_6bit", "csp6_mean"),
        ("test_acc_csp_4bit", "csp4_mean"),
    ):
        vals = [r[col] for r in rows if col in r]
        if vals:
            result[key] = float(np.mean(vals))
            result[key.replace("mean", "std")] = float(np.std(vals))
    return result


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

_COL_WIDTH = 16

def _print_table(
    subjects: list[int],
    stats: dict[int, dict],
    missing: list[int],
    dataset: str = "BNCI2014_001",
) -> None:
    """Print a formatted cross-subject accuracy table to stdout."""

    # Detect which optional columns are present in any subject
    has_lda  = any("lda_mean"  in stats.get(s, {}) for s in subjects)
    has_svm  = any("svm_mean"  in stats.get(s, {}) for s in subjects)
    has_ptq  = any("csp8_mean" in stats.get(s, {}) for s in subjects)

    # Build header / row format based on available columns
    # Columns: Subject | FP32 | [CSP-8b CSP-6b CSP-4b] | [LDA] | [SVM] | Folds
    col_defs: list[tuple[str, int]] = [("Subject", -8), ("FP32 (%)", 14)]
    if has_ptq:
        col_defs += [("CSP-8b", 8), ("CSP-6b", 8), ("CSP-4b", 8)]
    if has_lda:
        col_defs.append(("LDA (%)", 10))
    if has_svm:
        col_defs.append(("SVM (%)", 10))
    col_defs.append(("Folds", 6))

    def _fmt_row(values: list[str]) -> str:
        parts = []
        for v, (_, w) in zip(values, col_defs):
            parts.append(f"{v:>{abs(w)}}" if w > 0 else f"{v:<{abs(w)}}")
        return "  ".join(parts)

    sep_width = sum(abs(w) + 2 for _, w in col_defs) - 2
    sep = "-" * sep_width

    print()
    print(f"Cross-Subject Accuracy — FBCSP-SNN ({dataset})")
    print(sep)
    print(_fmt_row([c for c, _ in col_defs]))
    print(sep)

    fp32_vals: list[float] = []
    int8_vals: list[float] = []
    lda_vals:  list[float] = []
    svm_vals:  list[float] = []
    csp8_vals: list[float] = []
    csp6_vals: list[float] = []
    csp4_vals: list[float] = []

    def _pct(st: dict, key: str) -> str:
        return f"{st[key]*100:.1f}" if key in st else "N/A"

    for s in subjects:
        if s in missing:
            blanks = ["--"] * len(col_defs)
            blanks[0] = f"S{s}"
            print(_fmt_row(blanks))
            continue
        st = stats[s]
        fp32_str = f"{st['fp32_mean']*100:.1f}±{st['fp32_std']*100:.1f}"
        row = [f"S{s}", fp32_str]
        if has_ptq:
            row += [_pct(st, "csp8_mean"), _pct(st, "csp6_mean"), _pct(st, "csp4_mean")]
        if has_lda:
            row.append(f"{st['lda_mean']*100:.1f}±{st['lda_std']*100:.1f}" if "lda_mean" in st else "N/A")
        if has_svm:
            row.append(f"{st['svm_mean']*100:.1f}±{st['svm_std']*100:.1f}" if "svm_mean" in st else "N/A")
        row.append(str(st["n_folds"]))
        print(_fmt_row(row))

        fp32_vals.append(st["fp32_mean"])
        if not has_lda and not has_svm and "int8_mean" in st:
            int8_vals.append(st["int8_mean"])
        if "lda_mean"  in st: lda_vals.append(st["lda_mean"])
        if "svm_mean"  in st: svm_vals.append(st["svm_mean"])
        if "csp8_mean" in st: csp8_vals.append(st["csp8_mean"])
        if "csp6_mean" in st: csp6_vals.append(st["csp6_mean"])
        if "csp4_mean" in st: csp4_vals.append(st["csp4_mean"])

    print(sep)

    if fp32_vals:
        grand_fp32     = np.mean(fp32_vals)
        grand_fp32_std = np.std(fp32_vals)
        total_folds    = sum(stats[s]["n_folds"] for s in subjects if s not in missing)

        grand_row = ["MEAN", f"{grand_fp32*100:.1f}±{grand_fp32_std*100:.1f}"]
        if has_ptq:
            grand_row += [
                f"{np.mean(csp8_vals)*100:.1f}" if csp8_vals else "N/A",
                f"{np.mean(csp6_vals)*100:.1f}" if csp6_vals else "N/A",
                f"{np.mean(csp4_vals)*100:.1f}" if csp4_vals else "N/A",
            ]
        if has_lda:
            grand_row.append(f"{np.mean(lda_vals)*100:.1f}" if lda_vals else "N/A")
        if has_svm:
            grand_row.append(f"{np.mean(svm_vals)*100:.1f}" if svm_vals else "N/A")
        grand_row.append(str(total_folds))
        print(_fmt_row(grand_row))
        print(sep)

        print(f"  Baseline target : 64.8% (FP32, static bands, 22 CSP comps)")
        print(f"  Target          : 70.0%+")
        delta = grand_fp32 * 100 - 64.8
        sign  = "+" if delta >= 0 else ""
        print(f"  SNN FP32        : {sign}{delta:.1f} pp vs baseline")

        if has_ptq and csp8_vals:
            drop8 = grand_fp32 - np.mean(csp8_vals)
            drop6 = grand_fp32 - np.mean(csp6_vals)
            drop4 = grand_fp32 - np.mean(csp4_vals)
            print(f"  PTQ-CSP drop    : 8-bit {drop8*100:+.2f}pp  |  6-bit {drop6*100:+.2f}pp  |  4-bit {drop4*100:+.2f}pp")
        if lda_vals:
            delta_lda = np.mean(lda_vals) * 100 - 64.8
            print(f"  LDA vs baseline : {'+'if delta_lda>=0 else ''}{delta_lda:.1f} pp")
        if svm_vals:
            delta_svm = np.mean(svm_vals) * 100 - 64.8
            print(f"  SVM vs baseline : {'+'if delta_svm>=0 else ''}{delta_svm:.1f} pp")

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
    dataset: str = "BNCI2014_001",
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

    has_lda  = any("lda_mean"  in stats[s] for s in valid)
    has_svm  = any("svm_mean"  in stats[s] for s in valid)
    has_ptq  = any("csp8_mean" in stats[s] for s in valid)

    x = np.arange(len(valid))

    fp32_means = np.array([stats[s]["fp32_mean"] * 100 for s in valid])
    fp32_stds  = np.array([stats[s]["fp32_std"]  * 100 for s in valid])

    if has_ptq:
        # PTQ layout: FP32 | CSP-8b | CSP-6b | CSP-4b  (+ optional LDA / SVM)
        bar_specs: list[tuple[str, str, str, str]] = [
            # label, stat_key, color, text_color
            ("SNN FP32",  "fp32_mean",  "steelblue",   "navy"),
            ("CSP-8bit",  "csp8_mean",  "mediumseagreen", "darkgreen"),
            ("CSP-6bit",  "csp6_mean",  "goldenrod",    "saddlebrown"),
            ("CSP-4bit",  "csp4_mean",  "tomato",       "darkred"),
        ]
        if has_lda:
            bar_specs.append(("LDA", "lda_mean", "slategray", "black"))
        if has_svm:
            bar_specs.append(("SVM", "svm_mean", "mediumpurple", "indigo"))

        n_bars = len(bar_specs)
        width  = 0.8 / n_bars
        offsets = np.linspace(-(n_bars - 1) / 2 * width, (n_bars - 1) / 2 * width, n_bars)

        fig, ax = plt.subplots(figsize=(max(12, len(valid) * 1.8), 5))

        for off, (label, key, color, tcol) in zip(offsets, bar_specs):
            vals = np.array([stats[s].get(key, 0) * 100 for s in valid])
            stds = np.array([stats[s].get(key.replace("mean", "std"), 0) * 100 for s in valid])
            bars = ax.bar(x + off, vals, width, yerr=stds, capsize=2,
                          color=color, alpha=0.82, label=label,
                          error_kw={"elinewidth": 0.8, "ecolor": tcol})
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=5.5, color=tcol)

    elif has_lda or has_svm:
        # No PTQ: SNN / LDA / SVM
        n_bars = 1 + int(has_lda) + int(has_svm)
        width  = 0.8 / n_bars
        offsets_list: list[float] = []
        labels_bars: list[str] = ["SNN (FP32)"]
        colors: list[str] = ["steelblue"]
        if has_lda:
            offsets_list.append(-width if n_bars == 3 else -width * 1.5)
            labels_bars.append("LDA")
            colors.append("seagreen")
        if has_svm:
            offsets_list.append(width if n_bars == 3 else -width * 0.5)
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
            off = offsets_list[bar_idx]; bar_idx += 1
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
            off = offsets_list[bar_idx]
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
        f"FBCSP-SNN Cross-Subject Accuracy — {dataset}\n"
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
    p.add_argument(
        "--moabb-dataset", default="BNCI2014_001",
        help="Dataset name shown in table headers and chart title "
             "(default: BNCI2014_001)",
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

    _print_table(args.subjects, stats, missing, dataset=args.moabb_dataset)

    chart_path = (
        Path(args.output)
        if args.output
        else results_dir / "cross_subject_accuracy.png"
    )
    _plot_bar_chart(args.subjects, stats, chart_path,
                    baseline_fp32=args.baseline, dataset=args.moabb_dataset)


if __name__ == "__main__":
    main()
