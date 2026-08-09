#!/usr/bin/env python
"""Plot accuracy against bit-width for each parameter group quantised alone.

This is the figure that carries the diagnosis: with every other group left at
full precision, the Euclidean-Alignment whitener is the line that dives while
the rest stay flat. As a table the same data is a 25-cell grid nobody reads.

Aggregation matches the rest of the analysis: the mean over *subject* means,
not over raw folds, so a subject with missing folds cannot skew a point.

Usage
-----
::

    python plot_pergroup.py --quant-dir Results_quant --dataset BNCI2015_001
    python plot_pergroup.py --quant-dir Results_quant --dataset BNCI2015_001 --fused

Writes a PDF (for LaTeX) and a PNG (for quick viewing) sized for a single
column of the Springer ``svproc`` layout.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Display names and a fixed drawing order, so the legend reads front-end
# first and classifier last rather than alphabetically.
GROUP_STYLE: Dict[str, Tuple[str, str, str]] = {
    "ea":     ("EA whitener",   "#d62728", "o"),
    "csp":    ("CSP filters",   "#1f77b4", "s"),
    "znorm":  ("$z$-norm",      "#2ca02c", "^"),
    "snn_w":  ("SNN weights",   "#9467bd", "D"),
    "snn_b":  ("SNN biases",    "#ff7f0e", "v"),
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    """Read a sweep CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[int, float]], Optional[float]]:
    """Mean accuracy per (group, bits), averaging subject means.

    Parameters
    ----------
    rows : list of dict
        Rows of a per-group sweep CSV.

    Returns
    -------
    curves : dict
        ``group -> {bits: mean accuracy}``.
    fp32 : float or None
        The full-precision reference, same averaging.
    """
    # subject -> condition -> [acc over folds]
    nested: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nested[int(r["subject"])][r["condition"]].append(float(r["test_acc"]))

    subj_means = {
        s: {c: statistics.mean(v) for c, v in conds.items()}
        for s, conds in nested.items()
    }

    curves: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_cond: Dict[str, List[float]] = defaultdict(list)
    for means in subj_means.values():
        for cond, val in means.items():
            per_cond[cond].append(val)

    fp32: Optional[float] = None
    for cond, vals in per_cond.items():
        mean = statistics.mean(vals)
        if cond == "fp32_reference":
            fp32 = mean
            continue
        if "@" not in cond:
            continue
        group, bit_str = cond.split("@", 1)
        curves[group][int(bit_str.replace("bit", ""))] = mean

    return dict(curves), fp32


def make_figure(curves: Dict[str, Dict[int, float]], fp32: Optional[float],
                out_stem: Path, chance: float = 50.0) -> None:
    """Draw and save the figure."""
    fig, ax = plt.subplots(figsize=(4.4, 2.8))

    if fp32 is not None:
        ax.axhline(fp32, color="0.35", lw=0.9, ls="--", zorder=1)
        ax.text(1.0, fp32 + 0.8, f"full precision ({fp32:.1f}%)",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="0.35")

    ax.axhline(chance, color="0.65", lw=0.9, ls=":", zorder=1)
    ax.text(0.0, chance - 1.2, "chance", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=7, color="0.55")

    all_bits = sorted({b for c in curves.values() for b in c})
    xpos = {b: i for i, b in enumerate(all_bits)}

    for group, (label, colour, marker) in GROUP_STYLE.items():
        if group not in curves:
            continue                      # fused runs have no EA group
        pts = sorted(curves[group].items())
        ax.plot([xpos[b] for b, _ in pts], [a for _, a in pts],
                marker=marker, ms=4, lw=1.3, color=colour, label=label, zorder=3)

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([str(b) for b in all_bits])
    ax.set_xlim(-0.25, len(all_bits) - 0.75)
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 82)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    """Build the per-group figure from a sweep CSV."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quant-dir", default="Results_quant")
    ap.add_argument("--dataset", default="BNCI2015_001")
    ap.add_argument("--fused", action="store_true",
                    help="Plot the fused-front-end sweep instead of the split one.")
    ap.add_argument("--output-dir", default="figures")
    args = ap.parse_args()

    suffix = "_fused" if args.fused else ""
    csv_path = Path(args.quant_dir) / f"quant_sweep_{args.dataset}_per-group{suffix}.csv"
    if not csv_path.exists():
        raise SystemExit(f"not found: {csv_path}")

    curves, fp32 = aggregate(load_rows(csv_path))
    if not curves:
        raise SystemExit(f"no per-group conditions in {csv_path}")

    print(f"groups: {', '.join(sorted(curves))}")
    print(f"fp32 reference: {fp32:.2f}%" if fp32 is not None else "no fp32 row")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(curves, fp32, out_dir / f"pergroup_{args.dataset}{suffix}")


if __name__ == "__main__":
    main()
