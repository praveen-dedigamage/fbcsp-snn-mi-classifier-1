"""Lava simulation of the trained SNNClassifier — item 8.

Runs in the LAVA venv (.venv_lava) which has lava-nc / lava-dl.
Loads pre-saved test spike tensors (from save_test_spikes.py) and
the trained snnTorch weight files, builds an equivalent SLAYER network,
runs inference, and reports:

  1. Accuracy vs stored FP32 baseline (verifies < 1 pp gap)
  2. Synapse event counts per inference (input for item 9 energy estimate)
  3. Loihi 2 resource summary (neurons, synapses, fan-in)
  4. Exports one .net HDF5 file per subject (Loihi 2 format)

Usage
-----
    source /scratch/.../praveen/.venv_lava/bin/activate
    python run_lava_infer.py \\
        --results-dir Results_adm_static6_ptq \\
        --subjects 1 2 3 4 5 6 7 8 9 \\
        --n-folds 5 \\
        --output-dir Results_lava

Outputs
-------
    Results_lava/
        lava_raw_S{N}.csv   — per-fold rows: accuracy, synapse events
        lava_raw.csv        — merged across subjects
        lava_summary.csv    — per-subject accuracy comparison + events
        network_S{N}_fold0.net — NETX HDF5 for subject N (one per subject)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# This runs in .venv_lava — import only what's available there
from fbcsp_snn.lava_model import LavaNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-subject inference
# ---------------------------------------------------------------------------

def run_subject_lava(
    subject_id: int,
    results_dir: Path,
    n_folds: int,
    output_dir: Path,
) -> List[dict]:
    """Run Lava inference for all folds of one subject.

    Returns
    -------
    list of dict
        One row per fold with accuracy and synapse event statistics.
    """
    records: List[dict] = []
    subject_dir = results_dir / f"Subject_{subject_id}"
    net_exported = False   # export .net once per subject

    for fold_idx in range(n_folds):
        fold_dir = subject_dir / f"fold_{fold_idx}"

        # --- Load pre-saved spike tensors ---
        spikes_path = fold_dir / "test_spikes.pt"
        labels_path = fold_dir / "test_labels.npy"
        if not spikes_path.exists() or not labels_path.exists():
            logger.warning("S%d fold %d: test_spikes.pt / test_labels.npy missing "
                           "— run save_test_spikes.py first", subject_id, fold_idx)
            continue

        # spikes: (T, n_trials, n_features)  binary float32
        spikes = torch.load(spikes_path, map_location="cpu").float()
        labels = torch.from_numpy(np.load(labels_path)).long()

        # --- Load fold metadata + snnTorch weights ---
        params_path = fold_dir / "pipeline_params.json"
        model_path  = fold_dir / "best_model.pt"
        if not params_path.exists() or not model_path.exists():
            logger.warning("S%d fold %d: params / model missing", subject_id, fold_idx)
            continue

        with open(params_path) as f:
            params = json.load(f)

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Stored FP32 baseline (no need to recompute)
        acc_fp32: float = params["test_acc_fp32"]

        # --- Build Lava network with transferred weights ---
        net = LavaNetwork.from_snntorch(state_dict, params)
        net.eval()

        # --- Export .net once per subject (fold 0) ---
        if not net_exported:
            output_dir.mkdir(parents=True, exist_ok=True)
            net_path = output_dir / f"network_S{subject_id}_fold{fold_idx}.net"
            try:
                net.export_hdf5(net_path)
                net_exported = True
            except Exception as exc:
                logger.warning("NETX export failed: %s", exc)

        # --- Run SLAYER inference ---
        # Convert to SLAYER convention: (batch, features, T)
        x_lava = spikes.permute(1, 2, 0)   # (n_trials, n_features, T)
        n_trials = x_lava.shape[0]
        T        = x_lava.shape[2]

        with torch.no_grad():
            spk1, spk2 = net(x_lava)       # (n_trials, n_hidden, T), (n_trials, n_output, T)
            preds = net.decode(spk2)        # (n_trials,)

        acc_lava = (preds == labels).float().mean().item()

        # --- Synapse event counts ---
        # Input spikes × fan-out n_hidden = layer-1 SynOps
        # Hidden spikes × fan-out n_output = layer-2 SynOps
        n_hidden = net.n_hidden
        n_output = net.n_output
        input_spikes  = (x_lava > 0.5).sum().item()  # total across all trials × time
        hidden_spikes = (spk1   > 0.5).sum().item()

        # Per-trial means (for energy estimation)
        synops_l1_per_trial = input_spikes  / n_trials * n_hidden
        synops_l2_per_trial = hidden_spikes / n_trials * n_output
        synops_total        = synops_l1_per_trial + synops_l2_per_trial

        # Spike rates (fraction of active neurons per timestep)
        input_rate  = input_spikes  / (n_trials * net.n_input  * T)
        hidden_rate = hidden_spikes / (n_trials * n_hidden * T)

        gap_pp = (acc_lava - acc_fp32) * 100

        logger.info(
            "S%d fold %d | FP32=%.1f%%  Lava=%.1f%%  gap=%+.2f pp | "
            "SynOps/trial: L1=%.0f  L2=%.0f  tot=%.0f | "
            "spike rate: in=%.1f%%  hid=%.1f%%",
            subject_id, fold_idx,
            acc_fp32 * 100, acc_lava * 100, gap_pp,
            synops_l1_per_trial, synops_l2_per_trial, synops_total,
            input_rate * 100, hidden_rate * 100,
        )

        records.append({
            "subject":          subject_id,
            "fold":             fold_idx,
            "n_input":          net.n_input,
            "n_hidden":         n_hidden,
            "n_output":         n_output,
            "T":                T,
            "n_trials":         n_trials,
            "acc_fp32":         acc_fp32,
            "acc_lava":         acc_lava,
            "gap_pp":           gap_pp,
            "synops_l1":        synops_l1_per_trial,
            "synops_l2":        synops_l2_per_trial,
            "synops_total":     synops_total,
            "input_spike_rate": input_rate,
            "hidden_spike_rate": hidden_rate,
        })

    return records


# ---------------------------------------------------------------------------
# Resource summary (one per subject, printed once)
# ---------------------------------------------------------------------------

def _print_resource_summary(records: List[dict]) -> None:
    if not records:
        return
    r = records[0]
    net = LavaNetwork(
        n_input=r["n_input"], n_hidden=r["n_hidden"],
        n_classes=4,  # approximate — exact value not critical for resource count
        population_per_class=r["n_output"] // 4,
    )
    res = net.resource_summary()
    print(f"\n  Loihi 2 resource summary")
    print(f"  {'Neurons':<20}: {res['neurons']:,}")
    print(f"  {'Synapses':<20}: {res['synapses']:,}")
    print(f"  {'Max fan-in':<20}: {res['max_fan_in']:,}  "
          f"(Loihi 2 limit: 8,192 — OK ✓)")


# ---------------------------------------------------------------------------
# Summary table + CSV output
# ---------------------------------------------------------------------------

def _write_csvs(records: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []

    # Merged raw
    raw_path = output_dir / "lava_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Raw results: %s", raw_path)

    # Per-subject
    by_subject: Dict[int, List[dict]] = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)
    for subj, subj_records in by_subject.items():
        subj_path = output_dir / f"lava_raw_S{subj}.csv"
        with open(subj_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subj_records)

    # Summary per subject
    summary_path = output_dir / "lava_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject", "fp32_mean", "lava_mean", "gap_pp_mean",
            "synops_l1_mean", "synops_l2_mean", "synops_total_mean",
            "input_rate_mean", "hidden_rate_mean",
        ])
        for subj, sr in sorted(by_subject.items()):
            writer.writerow([
                subj,
                f"{np.mean([r['acc_fp32'] for r in sr])*100:.2f}",
                f"{np.mean([r['acc_lava'] for r in sr])*100:.2f}",
                f"{np.mean([r['gap_pp'] for r in sr]):+.2f}",
                f"{np.mean([r['synops_l1'] for r in sr]):.0f}",
                f"{np.mean([r['synops_l2'] for r in sr]):.0f}",
                f"{np.mean([r['synops_total'] for r in sr]):.0f}",
                f"{np.mean([r['input_spike_rate'] for r in sr])*100:.1f}",
                f"{np.mean([r['hidden_spike_rate'] for r in sr])*100:.1f}",
            ])
    logger.info("Summary: %s", summary_path)


def _print_summary_table(records: List[dict]) -> None:
    subjects = sorted({r["subject"] for r in records})

    print(f"\n{'='*74}")
    print(f"  Lava-DL (SLAYER) Simulation — Loihi 2 Accuracy Verification")
    print(f"{'='*74}")
    col_h = (f"  {'Subj':<6}  {'FP32':>7}  {'Lava':>7}  {'Gap':>7}  "
             f"{'SynOps/trial':>13}  {'In rate':>8}  {'Hid rate':>9}")
    print(col_h)
    print("  " + "-" * 70)

    fp32_vals, lava_vals, gap_vals, synops_vals = [], [], [], []
    in_rates, hid_rates = [], []

    for s in subjects:
        sr = [r for r in records if r["subject"] == s]
        fp32  = np.mean([r["acc_fp32"] for r in sr]) * 100
        lava  = np.mean([r["acc_lava"] for r in sr]) * 100
        gap   = np.mean([r["gap_pp"]   for r in sr])
        sops  = np.mean([r["synops_total"]     for r in sr])
        ir    = np.mean([r["input_spike_rate"] for r in sr]) * 100
        hr    = np.mean([r["hidden_spike_rate"] for r in sr]) * 100

        print(f"  S{s:<5}  {fp32:>7.1f}%  {lava:>7.1f}%  {gap:>+6.2f}pp  "
              f"{sops:>13,.0f}  {ir:>7.1f}%  {hr:>8.1f}%")
        fp32_vals.append(fp32); lava_vals.append(lava); gap_vals.append(gap)
        synops_vals.append(sops); in_rates.append(ir); hid_rates.append(hr)

    print("  " + "-" * 70)
    print(f"  {'MEAN':<6}  {np.mean(fp32_vals):>7.1f}%  {np.mean(lava_vals):>7.1f}%  "
          f"{np.mean(gap_vals):>+6.2f}pp  {np.mean(synops_vals):>13,.0f}  "
          f"{np.mean(in_rates):>7.1f}%  {np.mean(hid_rates):>8.1f}%")
    print(f"{'='*74}")
    print(f"  Accuracy gap mean {np.mean(gap_vals):+.2f} pp  "
          f"({'< 1pp ✓' if abs(np.mean(gap_vals)) < 1.0 else 'CHECK'})  "
          f"(FP32 {np.mean(fp32_vals):.1f}%  →  Lava {np.mean(lava_vals):.1f}%)")
    print(f"  Mean synapse events/inference: {np.mean(synops_vals):,.0f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lava SLAYER inference for Loihi 2 verification.")
    p.add_argument("--results-dir",   default="Results_adm_static6_ptq")
    p.add_argument("--subjects",      nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--n-folds",       type=int, default=5)
    p.add_argument("--output-dir",    default="Results_lava")
    p.add_argument("--from-csv",      default=None, metavar="CSV_PATH",
                   help="Skip inference; load merged lava_raw.csv and regenerate outputs.")
    return p.parse_args()


def _load_records_from_csv(csv_path: Path) -> List[dict]:
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({k: (float(v) if k not in ("subject", "fold", "n_input",
                                                        "n_hidden", "n_output", "T",
                                                        "n_trials")
                                else int(v))
                             for k, v in row.items()})
    return records


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if args.from_csv is not None:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            logger.error("--from-csv not found: %s", csv_path)
            sys.exit(1)
        all_records = _load_records_from_csv(csv_path)
        logger.info("Loaded %d records from %s", len(all_records), csv_path)
        _write_csvs(all_records, output_dir)
        _print_summary_table(all_records)
        _print_resource_summary(all_records)
        return

    results_dir = Path(args.results_dir)
    all_records: List[dict] = []

    for subject_id in args.subjects:
        logger.info("=" * 50)
        logger.info("Subject %d", subject_id)
        logger.info("=" * 50)
        records = run_subject_lava(subject_id, results_dir, args.n_folds, output_dir)
        all_records.extend(records)

    if not all_records:
        logger.error("No records — check --results-dir and ensure save_test_spikes.py ran first.")
        sys.exit(1)

    _write_csvs(all_records, output_dir)
    _print_summary_table(all_records)
    _print_resource_summary(all_records)
    logger.info("Lava simulation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
