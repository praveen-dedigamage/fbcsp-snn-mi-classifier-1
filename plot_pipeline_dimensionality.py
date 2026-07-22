"""Supplementary figure: real per-stage tensor dimensionality through the
FBCSP-SNN pipeline, for one real trained fold (Subject 1, Fold 0 --
the same fold used in Fig. 3's SNN raster).

Fig. 2 in the main paper illustrates the pipeline with ONE synthetic band
and ONE CSP class-pair comparison, kept deliberately small so the per-stage
signal transformation stays visually inspectable. That figure does not
convey how large the real feature space actually is: K=6 bands and
binom(4,2)=6 class pairs, each contributing 2m=8 dual-end CSP filters,
giving 288 raw CSP features before MIBIF selection. This script draws a
simple box-and-arrow diagram of tensor shapes at each stage, with every
number read directly from one real trained fold's pipeline_params.json
(not hand-typed), for a supplementary/appendix figure.

Usage:
    python plot_pipeline_dimensionality.py \\
        --results-dir Results_verify_Subject_1 --subject-id 1 --fold 0
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, title, shape_line, note=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.2, edgecolor="black", facecolor="white")
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, shape_line, ha="center", va="center",
            fontsize=10.5, family="monospace")
    if note:
        ax.text(x + w / 2, y - 0.09, note, ha="center", va="top", fontsize=9,
                style="italic", color="0.25")


def _arrow(ax, x0, y, x1, label=None):
    arr = FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14,
                           linewidth=1.2, color="black")
    ax.add_patch(arr)
    if label:
        ax.text((x0 + x1) / 2, y + 0.05, label, ha="center", va="bottom",
                fontsize=8.5, color="0.15")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="Results_verify_Subject_1")
    ap.add_argument("--subject-id", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-channels", type=int, default=22,
                     help="Raw EEG channel count (BNCI2014-001).")
    ap.add_argument("--output", default="figures/pipeline_dimensionality.png")
    args = ap.parse_args()

    fold_dir = os.path.join(args.results_dir, f"Subject_{args.subject_id}",
                             f"fold_{args.fold}")
    with open(os.path.join(fold_dir, "pipeline_params.json")) as f:
        params = json.load(f)

    n_ch = args.n_channels
    n_bands = len(params["bands"])
    m = params["csp_m"]
    n_classes = params["n_classes"]
    n_pairs = n_classes * (n_classes - 1) // 2
    n_csp_total = 2 * m * n_pairs * n_bands
    n_selected = params["n_input_features"]
    T = params["n_timesteps"]
    hidden = params["hidden_neurons"]
    pop = params["population_per_class"]
    n_out = n_classes * pop

    print(f"Read real shapes from {fold_dir}/pipeline_params.json: "
          f"n_ch={n_ch} n_bands={n_bands} m={m} n_pairs={n_pairs} "
          f"n_csp_total={n_csp_total} n_selected={n_selected} T={T} "
          f"hidden={hidden} n_out={n_out}")
    assert n_csp_total == 288, (
        f"Expected 288 raw CSP features (2*m*pairs*bands), got {n_csp_total} "
        f"-- pipeline_params.json values changed; update this script's "
        f"assumptions or investigate.")

    matplotlib.rcdefaults()
    fig, ax = plt.subplots(figsize=(14, 3.4))

    stages = [
        ("Raw EEG", f"{n_ch} ch x {T}", None),
        ("Filter Bank", f"{n_ch}x{n_bands} ch x {T}", f"K={n_bands} bands"),
        ("Pairwise CSP", f"{n_csp_total} feat x {T}",
         f"2m x {n_pairs} pairs x {n_bands} bands\n(m={m}, dual-end)"),
        ("Z-Norm", f"{n_csp_total} feat x {T}", "per-feature\nrescale only"),
        ("Spike Encoding", f"{n_csp_total} feat x {T}\n(binary)", "adaptive\nthreshold"),
        ("MIBIF Select", f"{n_selected} feat x {T}\n(binary)",
         f"{n_selected}/{n_csp_total} kept\n(adaptive MI\nthreshold)"),
        ("SNN", f"{hidden} hidden\n{n_out} output", f"P={pop}/class"),
    ]

    n = len(stages)
    box_w, box_h, gap = 1.55, 1.15, 0.55
    x = 0.2
    y = 0.55
    centers = []
    for title, shape_line, note in stages:
        _box(ax, x, y, box_w, box_h, title, shape_line, note)
        centers.append((x, x + box_w))
        x += box_w + gap

    arrow_labels = ["x6 bands", "dual-end CSP", "z-score", "delta encode",
                    "MIBIF (10% thr.)", "flatten"]
    arrow_y = y + box_h + 0.32
    for i in range(n - 1):
        _arrow(ax, centers[i][1], arrow_y, centers[i + 1][0],
               arrow_labels[i] if i < len(arrow_labels) else None)

    ax.set_xlim(0, x)
    ax.set_ylim(-0.15, arrow_y + 0.35)
    ax.axis("off")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(args.output.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {args.output} (and matching .pdf)")


if __name__ == "__main__":
    main()
