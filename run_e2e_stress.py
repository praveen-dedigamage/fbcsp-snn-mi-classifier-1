"""End-to-end hardware stress test — item 7.

Loads FP32-trained fold artifacts and evaluates with all three hardware
imperfections applied simultaneously:

  1. Gm-C filter mismatch  — σ% global process corner (correlated, all bands shift together)
  2. CSP weight precision  — N-bit symmetric per-tensor (analog crossbar)
  3. SNN weight precision  — INT8 symmetric per-tensor (neuromorphic chip)

Training is untouched (FP32).  This is post-hoc inference-only evaluation.

Usage
-----
    python run_e2e_stress.py \\
        --results-dir Results_adm_static6_ptq \\
        --subjects 1 2 3 4 5 6 7 8 9 \\
        --sigma 0.02 \\
        --csp-bits 4 \\
        --snn-bits 8 \\
        --n-draws 100 \\
        --n-folds 5 \\
        --output-dir Results_e2e_stress

Outputs
-------
    Results_e2e_stress/
        e2e_raw_S{N}.csv   — per-draw rows for subject N
        e2e_raw.csv        — merged across all subjects
        e2e_summary.csv    — per-subject breakdown (fp32 / quant-only / full-hw)
        e2e_stress.png     — stacked bar chart of accuracy drops
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).parent))

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.datasets import load_moabb
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser
from fbcsp_snn.quantization import quantize_csp_filters, quantize_model
from fbcsp_snn.training import evaluate_model

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Filter helpers  (identical to run_butterworth_mc.py)
# ---------------------------------------------------------------------------

def _band_sos(lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    lo_n = np.clip(lo / nyq, 1e-4, 1.0 - 1e-4)
    hi_n = np.clip(hi / nyq, 1e-4, 1.0 - 1e-4)
    return butter(order, [lo_n, hi_n], btype="bandpass", output="sos")


def _apply_filterbank(X: np.ndarray, sos_list: List[np.ndarray]) -> List[np.ndarray]:
    n_trials, n_channels, n_samples = X.shape
    X_2d = X.reshape(n_trials * n_channels, n_samples)
    out = []
    for sos in sos_list:
        filtered = sosfilt(sos, X_2d, axis=-1)
        if not np.isfinite(filtered).all():
            filtered = np.zeros_like(filtered)
        out.append(filtered.reshape(n_trials, n_channels, n_samples).astype(np.float32))
    return out


def _perturb_filterbank(
    bands: List[Tuple[float, float]],
    sfreq: float,
    sigma: float,
    rng: np.random.Generator,
    order: int = 4,
) -> List[np.ndarray]:
    """Correlated process-corner perturbation — one shared ε for all bands."""
    nyq = sfreq / 2.0
    eps = rng.standard_normal() * sigma
    perturbed = []
    for lo, hi in bands:
        lo_p = float(np.clip(lo * (1.0 + eps), 0.5, nyq - 1.0))
        hi_p = float(np.clip(hi * (1.0 + eps), lo_p + 0.5, nyq - 0.5))
        sos = butter(order, [lo_p / nyq, hi_p / nyq], btype="bandpass", output="sos")
        perturbed.append(sos)
    return perturbed


def _encode_spikes(
    proj: Dict,
    znorm: ZNormaliser,
    mibif: Optional[MIBIFSelector],
    encoder_type: str,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    X = np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1).astype(np.float32)
    X_norm = znorm.transform(X)
    X_t = torch.from_numpy(X_norm).to(DEVICE).permute(2, 0, 1)
    spikes = encode_tensor(X_t, base_thresh, adapt_inc, decay, encoder_type)
    if mibif is not None:
        spikes = mibif.transform(spikes)
    return spikes


# ---------------------------------------------------------------------------
# Per-subject stress test
# ---------------------------------------------------------------------------

def run_subject_e2e(
    subject_id: int,
    results_dir: Path,
    n_folds: int,
    X_test: np.ndarray,
    y_test_0: np.ndarray,
    sfreq: float,
    sigma: float,
    csp_bits: int,
    snn_bits: int,
    n_draws: int,
    seed: int,
) -> List[dict]:
    """Run end-to-end stress test for one subject.

    For each fold:
      1. Read FP32 baseline from stored pipeline_params.json (no recomputation).
      2. Compute quantization-only accuracy: INT{snn_bits} SNN + {csp_bits}-bit CSP,
         clean (nominal) filters.
      3. MC loop (n_draws): INT{snn_bits} SNN + {csp_bits}-bit CSP + σ% perturbed filters.

    Returns
    -------
    list of dict
        One row per (fold, draw).  Columns:
          subject, fold, draw,
          acc_fp32,          — stored FP32 baseline (per fold, same for all draws)
          acc_quant,         — INT8 SNN + 4-bit CSP, clean filters (per fold, same for all draws)
          acc_hw,            — INT8 SNN + 4-bit CSP + perturbed filter (per draw)
          drop_quant,        — acc_fp32 - acc_quant
          drop_filter,       — acc_quant - acc_hw
          drop_total,        — acc_fp32 - acc_hw
    """
    rng = np.random.default_rng(seed + subject_id)
    records: List[dict] = []
    subject_dir = results_dir / f"Subject_{subject_id}"

    for fold_idx in range(n_folds):
        fold_dir = subject_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            logger.warning("S%d fold %d: directory missing — skipping", subject_id, fold_idx)
            continue

        params_path = fold_dir / "pipeline_params.json"
        if not params_path.exists():
            logger.warning("S%d fold %d: pipeline_params.json missing", subject_id, fold_idx)
            continue

        with open(params_path) as f:
            params = json.load(f)

        # Read stored FP32 baseline — no need to recompute
        acc_fp32: float = params["test_acc_fp32"]

        # Load preprocessing artifacts
        try:
            with open(fold_dir / "csp_filters.pkl", "rb") as f:
                csp: PairwiseCSP = pickle.load(f)
            with open(fold_dir / "znorm.pkl", "rb") as f:
                znorm: ZNormaliser = pickle.load(f)
        except FileNotFoundError as exc:
            logger.warning("S%d fold %d: artifact missing (%s) — skipping", subject_id, fold_idx, exc)
            continue

        mibif: Optional[MIBIFSelector] = None
        mibif_path = fold_dir / "mibif.pkl"
        if mibif_path.exists():
            with open(mibif_path, "rb") as f:
                mibif = pickle.load(f)

        model_path = fold_dir / "best_model.pt"
        if not model_path.exists():
            logger.warning("S%d fold %d: best_model.pt missing — skipping", subject_id, fold_idx)
            continue

        n_classes    = params.get("n_classes", 4)
        n_input      = params["n_input_features"]
        encoder_type = params.get("encoder_type", "delta")
        base_thresh  = params.get("base_thresh", 0.001)
        adapt_inc    = params.get("adapt_inc", 0.6)
        decay        = params.get("decay", 0.95)
        bands: List[Tuple[float, float]] = [tuple(b) for b in params["bands"]]

        # Load FP32 model
        model = SNNClassifier(
            n_input=n_input,
            n_hidden=params.get("hidden_neurons", 64),
            n_classes=n_classes,
            population_per_class=params.get("population_per_class", 20),
            beta=params.get("beta", 0.95),
            dropout_prob=0.0,
        ).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Apply SNN weight quantization once per fold
        model_q = quantize_model(model, bits=snn_bits)

        # Apply CSP filter quantization once per fold
        csp_q = copy.deepcopy(csp)
        csp_q.filters_ = quantize_csp_filters(csp.filters_, bits=csp_bits)

        # --- Quantization-only baseline: INT8 SNN + 4-bit CSP + clean filters ---
        sos_clean = [_band_sos(lo, hi, sfreq) for lo, hi in bands]
        X_bands_clean = _apply_filterbank(X_test, sos_clean)
        proj_clean    = csp_q.transform(X_bands_clean)
        spikes_clean  = _encode_spikes(proj_clean, znorm, mibif, encoder_type,
                                       base_thresh, adapt_inc, decay)
        acc_quant, _ = evaluate_model(model_q, spikes_clean, y_test_0, DEVICE)

        logger.info(
            "S%d fold %d | FP32=%.1f%%  INT%d+CSP%db=%.1f%%  (quant drop %+.2f pp)",
            subject_id, fold_idx,
            acc_fp32 * 100,
            snn_bits, csp_bits, acc_quant * 100,
            (acc_quant - acc_fp32) * 100,
        )

        # --- MC loop: INT8 SNN + 4-bit CSP + perturbed filters ---
        draw_accs: List[float] = []
        for draw in range(n_draws):
            sos_p    = _perturb_filterbank(bands, sfreq, sigma, rng)
            X_bands_p = _apply_filterbank(X_test, sos_p)
            proj_p    = csp_q.transform(X_bands_p)
            spikes_p  = _encode_spikes(proj_p, znorm, mibif, encoder_type,
                                        base_thresh, adapt_inc, decay)
            acc_hw, _ = evaluate_model(model_q, spikes_p, y_test_0, DEVICE)
            draw_accs.append(acc_hw)
            records.append({
                "subject":      subject_id,
                "fold":         fold_idx,
                "draw":         draw,
                "acc_fp32":     acc_fp32,
                "acc_quant":    acc_quant,
                "acc_hw":       acc_hw,
                "drop_quant":   acc_fp32  - acc_quant,
                "drop_filter":  acc_quant - acc_hw,
                "drop_total":   acc_fp32  - acc_hw,
            })

        logger.info(
            "  σ=%4.1f%%  full-hw: %.1f%% ± %.1f%%  total drop: %+.2f pp",
            sigma * 100,
            np.mean(draw_accs) * 100,
            np.std(draw_accs) * 100,
            (acc_fp32 - np.mean(draw_accs)) * 100,
        )

    return records


# ---------------------------------------------------------------------------
# Summary + output
# ---------------------------------------------------------------------------

def _write_csvs(records: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "fold", "draw",
                  "acc_fp32", "acc_quant", "acc_hw",
                  "drop_quant", "drop_filter", "drop_total"]

    # Merged raw file
    raw_path = output_dir / "e2e_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Raw results: %s", raw_path)

    # Per-subject files (for SLURM array job merging)
    by_subject: Dict[int, List[dict]] = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)
    for subj, subj_records in by_subject.items():
        subj_path = output_dir / f"e2e_raw_S{subj}.csv"
        with open(subj_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subj_records)
        logger.info("Subject %d raw: %s", subj, subj_path)

    # Summary: per-subject means
    subjects = sorted({r["subject"] for r in records})
    summary_path = output_dir / "e2e_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "fp32_mean", "quant_mean", "hw_mean",
                         "drop_quant_mean", "drop_filter_mean", "drop_total_mean",
                         "drop_total_std", "drop_total_p95"])
        for s in subjects:
            sr = [r for r in records if r["subject"] == s]
            fp32   = np.mean([r["acc_fp32"]    for r in sr])
            quant  = np.mean([r["acc_quant"]   for r in sr])
            # hw: each draw is independent; mean/std over all draws × folds
            hw_arr = np.array([r["acc_hw"]     for r in sr])
            dq     = np.mean([r["drop_quant"]  for r in sr])
            df     = np.mean([r["drop_filter"] for r in sr])
            dt_arr = fp32 - hw_arr
            writer.writerow([
                s,
                f"{fp32*100:.2f}", f"{quant*100:.2f}", f"{hw_arr.mean()*100:.2f}",
                f"{dq*100:.2f}",   f"{df*100:.2f}",
                f"{dt_arr.mean()*100:.2f}",
                f"{dt_arr.std()*100:.2f}",
                f"{np.percentile(dt_arr, 95)*100:.2f}",
            ])
    logger.info("Summary: %s", summary_path)


def _print_summary_table(records: List[dict], sigma: float,
                         csp_bits: int, snn_bits: int) -> None:
    subjects = sorted({r["subject"] for r in records})

    header = (f"\n{'='*72}\n"
              f"  End-to-End Hardware Stress Test\n"
              f"  σ={sigma*100:.0f}% filter + {csp_bits}-bit CSP + INT{snn_bits} SNN\n"
              f"{'='*72}")
    print(header)

    col_h = f"  {'Subj':<6}  {'FP32':>8}  {'Quant-only':>11}  {'Full HW':>9}  "
    col_h += f"{'Δ-Quant':>8}  {'Δ-Filter':>9}  {'Δ-Total':>8}"
    print(col_h)
    print("  " + "-" * 68)

    fp32_vals, quant_vals, hw_vals = [], [], []
    dq_vals, df_vals, dt_vals = [], [], []

    for s in subjects:
        sr = [r for r in records if r["subject"] == s]
        fp32  = np.mean([r["acc_fp32"]    for r in sr]) * 100
        quant = np.mean([r["acc_quant"]   for r in sr]) * 100
        hw    = np.mean([r["acc_hw"]      for r in sr]) * 100
        dq    = np.mean([r["drop_quant"]  for r in sr]) * 100
        df    = np.mean([r["drop_filter"] for r in sr]) * 100
        dt    = np.mean([r["drop_total"]  for r in sr]) * 100
        print(f"  S{s:<5}  {fp32:>8.1f}%  {quant:>10.1f}%  {hw:>8.1f}%  "
              f"{dq:>+7.2f}pp  {df:>+8.2f}pp  {dt:>+7.2f}pp")
        fp32_vals.append(fp32); quant_vals.append(quant); hw_vals.append(hw)
        dq_vals.append(dq);     df_vals.append(df);       dt_vals.append(dt)

    print("  " + "-" * 68)
    gfp32  = np.mean(fp32_vals)
    gquant = np.mean(quant_vals)
    ghw    = np.mean(hw_vals)
    gdq    = np.mean(dq_vals)
    gdf    = np.mean(df_vals)
    gdt    = np.mean(dt_vals)
    print(f"  {'MEAN':<6}  {gfp32:>8.1f}%  {gquant:>10.1f}%  {ghw:>8.1f}%  "
          f"{gdq:>+7.2f}pp  {gdf:>+8.2f}pp  {gdt:>+7.2f}pp")
    print(f"\n  FP32 → Quantization only : {gdq:+.2f} pp")
    print(f"  Quantization → +Filter   : {gdf:+.2f} pp")
    print(f"  FP32 → Full hardware     : {gdt:+.2f} pp  ({gfp32:.1f}% → {ghw:.1f}%)")
    print(f"{'='*72}\n")


def _plot_e2e(records: List[dict], output_dir: Path,
              sigma: float, csp_bits: int, snn_bits: int) -> None:
    subjects = sorted({r["subject"] for r in records})
    fp32_arr  = np.array([np.mean([r["acc_fp32"]  for r in records if r["subject"] == s]) for s in subjects]) * 100
    quant_arr = np.array([np.mean([r["acc_quant"] for r in records if r["subject"] == s]) for s in subjects]) * 100
    hw_arr    = np.array([np.mean([r["acc_hw"]    for r in records if r["subject"] == s]) for s in subjects]) * 100
    hw_std    = np.array([np.std ([r["acc_hw"]    for r in records if r["subject"] == s]) for s in subjects]) * 100

    x = np.arange(len(subjects))
    xlabels = [f"S{s}" for s in subjects] + ["Mean"]

    fp32_arr  = np.append(fp32_arr,  np.mean(fp32_arr))
    quant_arr = np.append(quant_arr, np.mean(quant_arr))
    hw_mean   = np.append(hw_arr,    np.mean(hw_arr))
    hw_std    = np.append(hw_std,    np.mean(hw_std))
    x_all     = np.arange(len(xlabels))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x_all - 0.25, fp32_arr,  0.25, label="FP32 (trained)", color="steelblue",   alpha=0.85)
    ax.bar(x_all,        quant_arr, 0.25, label=f"INT{snn_bits} SNN + {csp_bits}-bit CSP (clean filters)",
           color="darkorange", alpha=0.85)
    ax.bar(x_all + 0.25, hw_mean,   0.25, label=f"Full HW (+ σ={sigma*100:.0f}% filter)",
           color="tomato", alpha=0.85, yerr=hw_std, capsize=3,
           error_kw={"elinewidth": 1.0})

    ax.set_xticks(x_all)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_xlabel("Subject", fontsize=11)
    ax.set_title(f"End-to-End Hardware Stress Test\n"
                 f"σ={sigma*100:.0f}% Gm-C filter + {csp_bits}-bit CSP + INT{snn_bits} SNN  "
                 f"(BNCI2014_001, 9 subjects, 5-fold)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = output_dir / "e2e_stress.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end hardware stress test: filter mismatch + CSP quant + SNN quant."
    )
    p.add_argument("--results-dir",   default="Results_adm_static6_ptq")
    p.add_argument("--subjects",      nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--n-folds",       type=int,   default=5)
    p.add_argument("--sigma",         type=float, default=0.02,
                   help="Filter mismatch σ (default: 0.02 = 2%%)")
    p.add_argument("--csp-bits",      type=int,   default=4,
                   help="CSP filter quantization bits (default: 4)")
    p.add_argument("--snn-bits",      type=int,   default=8,
                   help="SNN weight quantization bits (default: 8)")
    p.add_argument("--n-draws",       type=int,   default=100)
    p.add_argument("--output-dir",    default="Results_e2e_stress")
    p.add_argument("--moabb-dataset", default="BNCI2014_001")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--from-csv",      default=None, metavar="CSV_PATH",
                   help="Skip MC loop; load merged e2e_raw.csv and regenerate outputs only.")
    return p.parse_args()


def _load_records_from_csv(csv_path: Path) -> List[dict]:
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({
                "subject":      int(row["subject"]),
                "fold":         int(row["fold"]),
                "draw":         int(row["draw"]),
                "acc_fp32":     float(row["acc_fp32"]),
                "acc_quant":    float(row["acc_quant"]),
                "acc_hw":       float(row["acc_hw"]),
                "drop_quant":   float(row["drop_quant"]),
                "drop_filter":  float(row["drop_filter"]),
                "drop_total":   float(row["drop_total"]),
            })
    return records


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if args.from_csv is not None:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            logger.error("--from-csv path not found: %s", csv_path)
            sys.exit(1)
        all_records = _load_records_from_csv(csv_path)
        logger.info("Loaded %d records from %s", len(all_records), csv_path)
        _write_csvs(all_records, output_dir)
        _print_summary_table(all_records, args.sigma, args.csp_bits, args.snn_bits)
        _plot_e2e(all_records, output_dir, args.sigma, args.csp_bits, args.snn_bits)
        return

    results_dir = Path(args.results_dir)
    all_records: List[dict] = []

    for subject_id in args.subjects:
        logger.info("=" * 50)
        logger.info("Subject %d", subject_id)
        logger.info("=" * 50)

        _, _, X_test, y_test = load_moabb(args.moabb_dataset, subject_id)
        y_test_0 = y_test - 1
        sfreq = 250.0  # BNCI2014_001

        records = run_subject_e2e(
            subject_id  = subject_id,
            results_dir = results_dir,
            n_folds     = args.n_folds,
            X_test      = X_test,
            y_test_0    = y_test_0,
            sfreq       = sfreq,
            sigma       = args.sigma,
            csp_bits    = args.csp_bits,
            snn_bits    = args.snn_bits,
            n_draws     = args.n_draws,
            seed        = args.seed,
        )
        all_records.extend(records)

    if not all_records:
        logger.error("No records collected — check --results-dir and --subjects.")
        sys.exit(1)

    _write_csvs(all_records, output_dir)
    _print_summary_table(all_records, args.sigma, args.csp_bits, args.snn_bits)
    _plot_e2e(all_records, output_dir, args.sigma, args.csp_bits, args.snn_bits)
    logger.info("E2E stress test complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
