"""
Visualise 4-bit quantized CSP filter matrices for one subject / fold.

Usage (on Puhti or locally):
    python show_csp_quantized.py --results-dir Results_adm_static6_ptq \
                                  --subject 1 --fold 0 --bits 4

Prints:
  - Original float32 values for every (band, pair) matrix
  - Scale factor
  - Integer grid after 4-bit quantization
  - Unique integers used and their counts
  - Saves a heatmap PNG alongside the script
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Quantisation (mirrors fbcsp_snn/quantization.py)
# ---------------------------------------------------------------------------

def quantize_array(x: np.ndarray, bits: int):
    q_max = float(2 ** (bits - 1) - 1)
    abs_max = float(np.abs(x).max())
    if abs_max == 0.0:
        return x.copy(), 1.0, np.zeros_like(x, dtype=int)
    scale = abs_max / q_max
    q_int = np.round(x / scale).clip(-q_max, q_max).astype(int)
    q_float = (q_int * scale).astype(x.dtype)
    return q_float, scale, q_int


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="Results_adm_static6_ptq")
    ap.add_argument("--subject",     type=int, default=1)
    ap.add_argument("--fold",        type=int, default=0)
    ap.add_argument("--bits",        type=int, default=4)
    ap.add_argument("--band",        type=int, default=None,
                    help="Show only this band index (0-5). Default: all bands.")
    args = ap.parse_args()

    csp_path = Path(args.results_dir) / f"Subject_{args.subject}" / \
               f"fold_{args.fold}" / "csp_filters.pkl"

    if not csp_path.exists():
        print(f"ERROR: {csp_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(csp_path, "rb") as f:
        csp = pickle.load(f)

    filters = csp.filters_          # dict: (band_idx, pair) -> np.ndarray (22, 8)
    keys    = sorted(filters.keys())
    bands   = sorted({k[0] for k in keys})
    pairs   = sorted({k[1] for k in keys})

    if args.band is not None:
        bands = [b for b in bands if b == args.band]

    bits = args.bits
    q_max = 2 ** (bits - 1) - 1     # 7 for 4-bit

    print(f"\n{'='*70}")
    print(f"  CSP filter matrices — {bits}-bit quantisation")
    print(f"  Subject {args.subject}  Fold {args.fold}  |  "
          f"{len(keys)} matrices  |  q_max = ±{q_max}  ({2*q_max+1} levels)")
    print(f"{'='*70}\n")

    all_integers = []

    for band_idx in bands:
        for pair in pairs:
            key = (band_idx, pair)
            if key not in filters:
                continue
            W = filters[key]            # shape (22, 8)
            _, scale, W_int = quantize_array(W, bits)

            all_integers.append(W_int.ravel())

            unique, counts = np.unique(W_int, return_counts=True)

            print(f"  Band {band_idx}  Pair {pair}   "
                  f"shape {W.shape}   scale = {scale:.6f}")
            print(f"  FP32 range : [{W.min():+.4f}, {W.max():+.4f}]")
            print(f"  INT{bits} range: [{W_int.min():+d}, {W_int.max():+d}]   "
                  f"unique levels used: {len(unique)} / {2*q_max+1}")
            print()

            # Print the integer grid (rows = channels, cols = filters)
            header = "  ch\\filt  " + "".join(f"  f{c:<2}" for c in range(W_int.shape[1]))
            print(header)
            print("  " + "-" * (len(header) - 2))
            for ch in range(W_int.shape[0]):
                row = f"  ch{ch:02d}     " + \
                      "".join(f"  {v:+3d}" for v in W_int[ch])
                print(row)

            # Unique value histogram
            hist = "  Histogram: " + \
                   "  ".join(f"{v:+d}×{c}" for v, c in zip(unique, counts))
            print(hist)
            print()

    # -----------------------------------------------------------------------
    # Summary across all shown matrices
    # -----------------------------------------------------------------------
    all_int = np.concatenate(all_integers)
    unique_all, counts_all = np.unique(all_int, return_counts=True)

    print(f"{'='*70}")
    print(f"  Summary across {len(bands)} band(s) × {len(pairs)} pairs")
    print(f"  Total integers : {len(all_int)}")
    print(f"  Unique levels  : {len(unique_all)} / {2*q_max+1}")
    print(f"  Distribution   : min={all_int.min():+d}  max={all_int.max():+d}  "
          f"mean={all_int.mean():+.2f}  std={all_int.std():.2f}")
    print(f"{'='*70}\n")

    # -----------------------------------------------------------------------
    # Heatmap — one grid per band showing all 6 pair matrices side by side
    # -----------------------------------------------------------------------
    n_bands_show = len(bands)
    n_pairs      = len(pairs)
    n_channels   = list(filters.values())[0].shape[0]
    n_filters    = list(filters.values())[0].shape[1]

    fig_h = max(4, n_channels * 0.3 * n_bands_show)
    fig_w = n_pairs * (n_filters * 0.5 + 0.5)

    fig, axes = plt.subplots(
        n_bands_show, n_pairs,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.suptitle(
        f"CSP filter matrices — {bits}-bit integers\n"
        f"Subject {args.subject}  Fold {args.fold}  "
        f"(range ±{q_max}, {2*q_max+1} levels)",
        fontsize=10,
    )

    for bi, band_idx in enumerate(bands):
        for pi, pair in enumerate(pairs):
            ax = axes[bi][pi]
            key = (band_idx, pair)
            if key not in filters:
                ax.axis("off")
                continue
            W = filters[key]
            _, _, W_int = quantize_array(W, bits)

            im = ax.imshow(W_int, aspect="auto", cmap="RdBu_r",
                           vmin=-q_max, vmax=q_max, interpolation="nearest")
            ax.set_title(f"B{band_idx} P{pair[0]}{pair[1]}", fontsize=7)
            ax.set_xlabel("Filter", fontsize=6)
            if pi == 0:
                ax.set_ylabel(f"Band {band_idx}\nChannel", fontsize=6)
            ax.tick_params(labelsize=5)
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_png = Path(f"csp_{bits}bit_S{args.subject}_fold{args.fold}.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {out_png}")


if __name__ == "__main__":
    main()
