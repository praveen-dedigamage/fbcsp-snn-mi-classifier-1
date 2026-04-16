"""Butterworth filter coefficient Monte Carlo sensitivity analysis.

Simulates Gm-C analog filter manufacturing variation by perturbing each SOS
coefficient with multiplicative Gaussian noise (coeff × (1 + ε), ε ~ N(0,σ)).
Runs N draws per sigma level per fold, records accuracy, and reports the
distribution. Validates the analog front-end claim for the paper.

Usage
-----
    python run_butterworth_mc.py \\
        --results-dir Results_adm_static6_ptq \\
        --subjects 1 2 3 4 5 6 7 8 9 \\
        --sigmas 0.01 0.02 0.05 \\
        --n-draws 100 \\
        --output-dir Results_butterworth_mc \\
        --moabb-dataset BNCI2014_001

Outputs
-------
    Results_butterworth_mc/
        mc_raw.csv        — per-draw rows: subject, fold, sigma, draw, acc, drop
        mc_summary.csv    — mean ± std per subject per sigma
        butterworth_mc.png — violin plot of accuracy distributions
"""

from __future__ import annotations

import argparse
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
from fbcsp_snn.training import evaluate_model

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _band_sos(lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    """Return Butterworth bandpass SOS array for one frequency band."""
    nyq = sfreq / 2.0
    lo_n = np.clip(lo / nyq, 1e-4, 1.0 - 1e-4)
    hi_n = np.clip(hi / nyq, 1e-4, 1.0 - 1e-4)
    return butter(order, [lo_n, hi_n], btype="bandpass", output="sos")


def _apply_filterbank(
    X: np.ndarray,
    sos_list: List[np.ndarray],
) -> List[np.ndarray]:
    """Apply a list of SOS arrays to X, one per band.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_trials, n_channels, n_samples)``.
    sos_list : list of np.ndarray
        One SOS array per band.

    Returns
    -------
    list of np.ndarray
        One filtered array per band, each shape ``(n_trials, n_channels, n_samples)``.
    """
    n_trials, n_channels, n_samples = X.shape
    X_2d = X.reshape(n_trials * n_channels, n_samples)
    out = []
    for sos in sos_list:
        filtered = sosfilt(sos, X_2d, axis=-1)
        out.append(filtered.reshape(n_trials, n_channels, n_samples).astype(np.float32))
    return out


def _perturb_sos(
    sos_list: List[np.ndarray],
    sigma: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Return a new list of SOS arrays with multiplicative Gaussian noise.

    Each coefficient c is replaced by c × (1 + ε), ε ~ N(0, sigma).
    Covers all 6 entries per section (b0, b1, b2, a0, a1, a2), modelling
    independent process variation in every Gm-C cell.

    Parameters
    ----------
    sos_list : list of np.ndarray
        Clean SOS arrays, one per band.
    sigma : float
        Relative noise standard deviation (e.g. 0.01 = 1%).
    rng : np.random.Generator
        NumPy RNG for reproducibility.

    Returns
    -------
    list of np.ndarray
        Perturbed SOS arrays, same structure as input.
    """
    return [
        sos * (1.0 + rng.standard_normal(sos.shape) * sigma)
        for sos in sos_list
    ]


def _encode_spikes(
    proj: Dict,
    znorm: ZNormaliser,
    mibif: Optional[MIBIFSelector],
    encoder_type: str,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    """CSP projection dict → spike tensor ready for SNN inference.

    Parameters
    ----------
    proj : dict
        Output of ``PairwiseCSP.transform``.
    znorm : ZNormaliser
        Fitted normaliser (from training fold).
    mibif : MIBIFSelector or None
        Fitted feature selector (from training fold).
    encoder_type : str
        ``'delta'`` or ``'adm'``.
    base_thresh, adapt_inc, decay : float
        Encoder hyperparameters.

    Returns
    -------
    torch.Tensor
        Binary spikes ``(T, n_trials, n_features)``.
    """
    X = np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1).astype(np.float32)
    X_norm = znorm.transform(X)                              # (n_trials, n_feat, n_samples)
    X_t = torch.from_numpy(X_norm).to(DEVICE).permute(2, 0, 1)  # (T, B, F)
    spikes = encode_tensor(X_t, base_thresh, adapt_inc, decay, encoder_type)
    if mibif is not None:
        spikes = mibif.transform(spikes)
    return spikes


# ---------------------------------------------------------------------------
# Per-subject Monte Carlo
# ---------------------------------------------------------------------------

def run_subject_mc(
    subject_id: int,
    results_dir: Path,
    n_folds: int,
    X_test: np.ndarray,
    y_test_0: np.ndarray,
    sfreq: float,
    sigmas: List[float],
    n_draws: int,
    seed: int,
) -> List[dict]:
    """Run Monte Carlo sensitivity analysis for one subject.

    Parameters
    ----------
    subject_id : int
        1-indexed subject ID.
    results_dir : Path
        Root results directory containing ``Subject_N/fold_K/`` artifacts.
    n_folds : int
        Number of folds to iterate.
    X_test : np.ndarray
        Test EEG ``(n_trials, n_channels, n_samples)``.
    y_test_0 : np.ndarray
        0-indexed test labels.
    sfreq : float
        Sampling frequency in Hz.
    sigmas : list of float
        Noise levels to evaluate.
    n_draws : int
        Monte Carlo draws per sigma per fold.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list of dict
        One row per (fold, sigma, draw).
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
            logger.warning("S%d fold %d: pipeline_params.json missing — skipping", subject_id, fold_idx)
            continue

        with open(params_path) as f:
            params = json.load(f)

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

        # Restore fold hyperparameters from saved params
        n_classes    = params.get("n_classes", 4)
        n_input      = params["n_input_features"]
        encoder_type = params.get("encoder_type", "delta")
        base_thresh  = params.get("base_thresh", 0.001)
        adapt_inc    = params.get("adapt_inc", 0.6)
        decay        = params.get("decay", 0.95)
        bands: List[Tuple[float, float]] = [tuple(b) for b in params["bands"]]

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

        # Precompute clean SOS once per fold
        sos_clean = [_band_sos(lo, hi, sfreq) for lo, hi in bands]

        # Baseline accuracy using clean (unperturbed) filters
        X_bands_clean = _apply_filterbank(X_test, sos_clean)
        proj_clean    = csp.transform(X_bands_clean)
        spikes_clean  = _encode_spikes(proj_clean, znorm, mibif, encoder_type,
                                       base_thresh, adapt_inc, decay)
        baseline_acc, _ = evaluate_model(model, spikes_clean, y_test_0, DEVICE)

        logger.info(
            "S%d fold %d | baseline %.1f%% | encoder=%s | n_input=%d | bands=%d",
            subject_id, fold_idx, baseline_acc * 100, encoder_type, n_input, len(bands),
        )

        # Monte Carlo loop
        for sigma in sigmas:
            draw_accs: List[float] = []
            for draw in range(n_draws):
                sos_perturbed = _perturb_sos(sos_clean, sigma, rng)
                X_bands_p = _apply_filterbank(X_test, sos_perturbed)
                proj_p    = csp.transform(X_bands_p)
                spikes_p  = _encode_spikes(proj_p, znorm, mibif, encoder_type,
                                           base_thresh, adapt_inc, decay)
                acc, _ = evaluate_model(model, spikes_p, y_test_0, DEVICE)
                draw_accs.append(acc)
                records.append({
                    "subject":      subject_id,
                    "fold":         fold_idx,
                    "sigma":        sigma,
                    "draw":         draw,
                    "accuracy":     acc,
                    "baseline_acc": baseline_acc,
                    "acc_drop":     baseline_acc - acc,
                })

            logger.info(
                "  σ=%4.1f%%  acc: %.1f%% ± %.1f%%  drop: %+.2f%%",
                sigma * 100,
                np.mean(draw_accs) * 100,
                np.std(draw_accs) * 100,
                (baseline_acc - np.mean(draw_accs)) * 100,
            )

    return records


# ---------------------------------------------------------------------------
# Summary + plotting
# ---------------------------------------------------------------------------

def _write_csvs(records: List[dict], output_dir: Path) -> None:
    """Write mc_raw.csv and mc_summary.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw draws
    raw_path = output_dir / "mc_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "fold", "sigma", "draw",
                                               "accuracy", "baseline_acc", "acc_drop"])
        writer.writeheader()
        writer.writerows(records)
    logger.info("Raw results: %s", raw_path)

    # Summary: mean ± std per subject per sigma
    from collections import defaultdict
    # key: (subject, sigma) → list of accuracy values
    grouped: Dict[Tuple, List[float]] = defaultdict(list)
    baseline: Dict[Tuple, float] = {}
    for r in records:
        key = (r["subject"], r["sigma"])
        grouped[key].append(r["accuracy"])
        baseline[(r["subject"], r["sigma"])] = r["baseline_acc"]

    summary_path = output_dir / "mc_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "sigma", "baseline_acc", "mean_acc",
                         "std_acc", "mean_drop", "p5_drop", "p95_drop"])
        for (subj, sigma), accs in sorted(grouped.items()):
            arr = np.array(accs)
            bl  = baseline[(subj, sigma)]
            drops = bl - arr
            writer.writerow([
                subj, f"{sigma:.3f}",
                f"{bl*100:.2f}",
                f"{arr.mean()*100:.2f}",
                f"{arr.std()*100:.2f}",
                f"{drops.mean()*100:.2f}",
                f"{np.percentile(drops, 5)*100:.2f}",
                f"{np.percentile(drops, 95)*100:.2f}",
            ])
    logger.info("Summary: %s", summary_path)


def _plot_mc(records: List[dict], output_dir: Path, sigmas: List[float]) -> None:
    """Violin plot of accuracy drop distributions per sigma level."""
    from collections import defaultdict

    # Aggregate drops across all subjects and folds, per sigma
    drops_by_sigma: Dict[float, List[float]] = defaultdict(list)
    baseline_mean: Dict[float, float] = defaultdict(list)  # type: ignore[assignment]
    for r in records:
        drops_by_sigma[r["sigma"]].append(r["acc_drop"] * 100)
        baseline_mean[r["sigma"]] = r["baseline_acc"] * 100  # approximate

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin plot of accuracy drop
    ax = axes[0]
    sigma_labels = [f"{s*100:.0f}%" for s in sigmas]
    data_for_violin = [drops_by_sigma[s] for s in sigmas]
    parts = ax.violinplot(data_for_violin, positions=range(len(sigmas)),
                          showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)
    ax.axhline(0, color="black", lw=1.0, ls="--", label="No drop")
    ax.axhline(1.0, color="red", lw=0.8, ls=":", label="1 pp threshold")
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(sigma_labels)
    ax.set_xlabel("Filter coefficient noise σ", fontsize=11)
    ax.set_ylabel("Accuracy drop (pp)", fontsize=11)
    ax.set_title("Gm-C Filter Mismatch — Accuracy Drop Distribution\n"
                 "(all subjects × folds × draws)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # Right: per-subject mean drop at each sigma
    ax2 = axes[1]
    subjects = sorted({r["subject"] for r in records})
    x = np.arange(len(subjects))
    colors = ["steelblue", "darkorange", "tomato"]
    width = 0.8 / len(sigmas)
    offsets = np.linspace(-(len(sigmas)-1)/2*width, (len(sigmas)-1)/2*width, len(sigmas))

    for off, sigma, color, label in zip(offsets, sigmas, colors, sigma_labels):
        subj_means = []
        subj_stds  = []
        for s in subjects:
            d = [r["acc_drop"] * 100 for r in records
                 if r["subject"] == s and r["sigma"] == sigma]
            subj_means.append(np.mean(d) if d else 0.0)
            subj_stds.append(np.std(d) if d else 0.0)
        ax2.bar(x + off, subj_means, width, yerr=subj_stds, capsize=2,
                color=color, alpha=0.8, label=f"σ={label}",
                error_kw={"elinewidth": 0.8})

    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{s}" for s in subjects])
    ax2.set_xlabel("Subject", fontsize=11)
    ax2.set_ylabel("Mean accuracy drop (pp)", fontsize=11)
    ax2.set_title("Per-Subject Accuracy Drop vs Filter Mismatch", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", ls=":", alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / "butterworth_mc.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved: %s", out_path)


def _print_summary_table(records: List[dict], sigmas: List[float]) -> None:
    """Print a compact per-subject × sigma accuracy drop table."""
    subjects = sorted({r["subject"] for r in records})
    sigma_labels = [f"σ={s*100:.0f}%_drop" for s in sigmas]

    header = f"  {'Subj':<6}" + "".join(f" {lb:>12}" for lb in sigma_labels) + f"  {'Baseline':>9}"
    sep = "-" * len(header)
    print()
    print("  Butterworth Monte Carlo — Mean Accuracy Drop (pp) ± std")
    print(sep)
    print(header)
    print(sep)

    all_drops: Dict[float, List[float]] = {s: [] for s in sigmas}
    all_baselines: List[float] = []

    for subj in subjects:
        row = f"  S{subj:<5}"
        subj_recs = [r for r in records if r["subject"] == subj]
        baselines = list({(r["fold"], r["sigma"]): r["baseline_acc"]
                          for r in subj_recs}.values())
        bl_mean = np.mean(baselines) * 100 if baselines else 0.0
        all_baselines.append(bl_mean)

        for sigma in sigmas:
            drops = [r["acc_drop"] * 100 for r in subj_recs if r["sigma"] == sigma]
            if drops:
                m, s_ = np.mean(drops), np.std(drops)
                row += f" {m:>+6.2f}±{s_:.2f}"
                all_drops[sigma].extend(drops)
            else:
                row += f" {'N/A':>12}"
        row += f"  {bl_mean:>8.1f}%"
        print(row)

    print(sep)
    grand_row = f"  {'MEAN':<6}"
    for sigma in sigmas:
        d = all_drops[sigma]
        grand_row += f" {np.mean(d):>+6.2f}±{np.std(d):.2f}" if d else f" {'N/A':>12}"
    grand_row += f"  {np.mean(all_baselines):>8.1f}%"
    print(grand_row)
    print(sep)
    print()

    # Verdict per sigma
    for sigma in sigmas:
        d = all_drops[sigma]
        if d:
            worst = np.percentile(d, 95)
            verdict = "PASS (<1pp)" if worst < 1.0 else f"MARGINAL ({worst:.2f}pp p95)" if worst < 2.0 else f"FAIL ({worst:.2f}pp p95)"
            print(f"  σ={sigma*100:.0f}%  mean drop: {np.mean(d):+.2f}pp  p95 drop: {worst:.2f}pp  → {verdict}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Butterworth SOS coefficient Monte Carlo for Gm-C mismatch validation."
    )
    p.add_argument("--results-dir", default="Results",
                   help="Root results directory with trained fold artifacts.")
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.01, 0.02, 0.05],
                   help="Relative noise σ levels (default: 0.01 0.02 0.05).")
    p.add_argument("--n-draws", type=int, default=100,
                   help="Monte Carlo draws per fold per sigma (default: 100).")
    p.add_argument("--output-dir", default="Results_butterworth_mc")
    p.add_argument("--moabb-dataset", default="BNCI2014_001")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    all_records: List[dict] = []

    for subject_id in args.subjects:
        logger.info("=" * 50)
        logger.info("Subject %d", subject_id)
        logger.info("=" * 50)

        # Load test data once per subject
        _, _, X_test, y_test = load_moabb(args.moabb_dataset, subject_id)
        y_test_0 = y_test - 1
        sfreq = 250.0  # BNCI2014_001

        records = run_subject_mc(
            subject_id  = subject_id,
            results_dir = results_dir,
            n_folds     = args.n_folds,
            X_test      = X_test,
            y_test_0    = y_test_0,
            sfreq       = sfreq,
            sigmas      = args.sigmas,
            n_draws     = args.n_draws,
            seed        = args.seed,
        )
        all_records.extend(records)

    if not all_records:
        logger.error("No records collected — check results_dir and subject list.")
        sys.exit(1)

    _write_csvs(all_records, output_dir)
    _print_summary_table(all_records, args.sigmas)
    _plot_mc(all_records, output_dir, args.sigmas)

    logger.info("Monte Carlo complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
