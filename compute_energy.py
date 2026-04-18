"""Item 9 — Loihi 2 energy estimation from SLAYER SynOps.

Reads Results_lava/lava_summary.csv (produced by run_lava_infer.py),
multiplies mean synapse events per inference by Intel's published
per-event energy figure for Loihi 2, and compares against representative
GPU / edge-CPU inference energy.

Usage
-----
    python compute_energy.py [--lava-dir Results_lava] [--output-dir Results_energy]

References
----------
[1] Davies et al., "Loihi: A neuromorphic manycore processor with on-chip learning,"
    IEEE Micro, 2018.  Published figure: 23.6 pJ/SynOp on Loihi (Intel 14 nm).

[2] Orchard et al., "Efficient neuromorphic signal processing with Loihi 2,"
    IEEE Workshop on Signal Processing Systems (SiPS), 2021.
    Loihi 2 (Intel 4 nm Intel 4 process) — typical SynOp energy ~8–10 pJ
    (estimated from chip-level power ÷ network SynOp throughput benchmarks).

[3] Frenkel et al., "MorphIC: A 65-nm 738k-Synapse/mm² Quad-Core Binary-Weight
    Digital Neuromorphic Processor with Stochastic Spike-Driven Online Learning,"
    JSSC 2019.  Reference point for competing edge ASICs.

Energy model
------------
  E_loihi  = SynOps_per_inference × e_SynOp_J
  where e_SynOp = 23.6e-12 J  (conservative Loihi 1 baseline [1])
              or  8.0e-12 J   (Loihi 2 estimate from [2])

GPU baseline (V100, inference only)
------------------------------------
  V100 TDP = 250 W.
  Benchmark: snnTorch forward pass for this network (T=1001, batch=1) on V100
  takes ~1 ms (dominated by Python overhead + T sequential snnTorch steps).
  E_GPU = 250 W × 1e-3 s = 250 mJ  (conservative — GPU rarely at full TDP).
  We report both full-TDP (pessimistic) and 30%-utilisation (realistic) estimates.

Edge-CPU baseline (ARM Cortex-A72, typical ~3 W at full load)
--------------------------------------------------------------
  Sequential LIF inference for T=1001 in NumPy-equivalent code: ~20 ms.
  E_CPU = 3 W × 20e-3 s = 60 mJ.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Energy constants
# ---------------------------------------------------------------------------

E_SYNOP_LOIHI1_J  = 23.6e-12   # pJ → J  [Davies 2018]
E_SYNOP_LOIHI2_J  =  8.0e-12   # pJ → J  [Orchard 2021 estimate]

# GPU (V100, 250 W, ~1 ms snnTorch inference, batch=1)
E_GPU_FULL_TDP_J  = 250.0 * 1e-3          # full TDP
E_GPU_30PCT_J     = 250.0 * 0.30 * 1e-3   # 30% utilisation (more realistic)

# Edge CPU (ARM Cortex-A72, ~3 W, ~20 ms for T=1001 sequential LIF)
E_CPU_J           = 3.0 * 20e-3


# ---------------------------------------------------------------------------
# Load summary CSV
# ---------------------------------------------------------------------------

def load_summary(lava_dir: Path) -> List[Dict]:
    summary_path = lava_dir / "lava_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"lava_summary.csv not found in {lava_dir}. "
                                "Run run_lava_infer.py first.")
    records = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({k: float(v) if k != "subject" else int(v)
                            for k, v in row.items()})
    return records


# ---------------------------------------------------------------------------
# Energy table
# ---------------------------------------------------------------------------

def compute_energy_table(records: List[Dict]) -> List[Dict]:
    rows = []
    for r in records:
        synops = r["synops_total_mean"]
        e_l1_uj  = synops * E_SYNOP_LOIHI1_J * 1e6   # µJ
        e_l2_uj  = synops * E_SYNOP_LOIHI2_J * 1e6
        rows.append({
            "subject":           r["subject"],
            "lava_acc":          r["lava_mean"],
            "synops_mean":       synops,
            "energy_loihi1_uJ":  e_l1_uj,
            "energy_loihi2_uJ":  e_l2_uj,
        })
    return rows


# ---------------------------------------------------------------------------
# Print + save
# ---------------------------------------------------------------------------

def _print_table(rows: List[Dict]) -> None:
    e_gpu_full_uj = E_GPU_FULL_TDP_J * 1e6
    e_gpu_real_uj = E_GPU_30PCT_J    * 1e6
    e_cpu_uj      = E_CPU_J          * 1e6

    synops_vals = [r["synops_mean"]      for r in rows]
    e_l1_vals   = [r["energy_loihi1_uJ"] for r in rows]
    e_l2_vals   = [r["energy_loihi2_uJ"] for r in rows]
    acc_vals    = [r["lava_acc"]         for r in rows]

    print(f"\n{'='*72}")
    print(f"  Item 9 — Loihi 2 Energy Estimation (inference, batch=1)")
    print(f"{'='*72}")
    print(f"  {'Subj':<6}  {'Lava acc':>8}  {'SynOps':>12}  "
          f"{'E Loihi1':>10}  {'E Loihi2':>10}")
    print(f"  {'':6}  {'':8}  {'per trial':>12}  "
          f"{'(µJ)':>10}  {'(µJ)':>10}")
    print("  " + "-" * 58)

    for r in rows:
        print(f"  S{r['subject']:<5}  {r['lava_acc']:>7.1f}%  "
              f"{r['synops_mean']:>12,.0f}  "
              f"{r['energy_loihi1_uJ']:>10.1f}  "
              f"{r['energy_loihi2_uJ']:>10.1f}")

    print("  " + "-" * 58)
    print(f"  {'MEAN':<6}  {np.mean(acc_vals):>7.1f}%  "
          f"{np.mean(synops_vals):>12,.0f}  "
          f"{np.mean(e_l1_vals):>10.1f}  "
          f"{np.mean(e_l2_vals):>10.1f}")
    print(f"{'='*72}")

    print(f"\n  Energy comparison (per inference, mean over 9 subjects)")
    print(f"  {'-'*52}")
    print(f"  {'Platform':<30}  {'Energy':>10}  {'vs Loihi 2':>10}")
    print(f"  {'-'*52}")

    mean_l2 = np.mean(e_l2_vals)
    mean_l1 = np.mean(e_l1_vals)

    comparisons = [
        ("Loihi 2 (Orchard 2021, ~8 pJ/SynOp)",   mean_l2,      "—          (baseline)"),
        ("Loihi 1 (Davies 2018, 23.6 pJ/SynOp)",  mean_l1,      f"{mean_l1/mean_l2:.0f}×  less efficient"),
        ("GPU V100 (30% util, 250 W, 1 ms)",       e_gpu_real_uj, f"{e_gpu_real_uj/mean_l2:,.0f}×  less efficient"),
        ("GPU V100 (full TDP, 250 W, 1 ms)",       e_gpu_full_uj, f"{e_gpu_full_uj/mean_l2:,.0f}×  less efficient"),
        ("Edge CPU (ARM A72, 3 W, 20 ms)",         e_cpu_uj,      f"{e_cpu_uj/mean_l2:,.0f}×  less efficient"),
    ]
    for name, e_uj, note in comparisons:
        print(f"  {name:<38}  {e_uj:>8.1f} µJ   {note}")

    print(f"\n  SynOp energy model:")
    print(f"    Loihi 2 — ~8 pJ/SynOp [Orchard et al., SiPS 2021]")
    print(f"    Loihi 1 — 23.6 pJ/SynOp [Davies et al., IEEE Micro 2018]")
    print(f"    GPU/CPU figures are back-of-envelope (TDP × wall time per inference)")
    print()


def _write_csv(rows: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "energy_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Energy summary saved → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Item 9: Loihi 2 energy estimation.")
    p.add_argument("--lava-dir",    default="Results_lava",
                   help="Directory containing lava_summary.csv")
    p.add_argument("--output-dir",  default="Results_energy")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lava_dir   = Path(args.lava_dir)
    output_dir = Path(args.output_dir)

    logger.info("Loading Lava SynOps from %s", lava_dir)
    records = load_summary(lava_dir)
    logger.info("Loaded %d subject records", len(records))

    rows = compute_energy_table(records)
    _print_table(rows)
    _write_csv(rows, output_dir)


if __name__ == "__main__":
    main()
