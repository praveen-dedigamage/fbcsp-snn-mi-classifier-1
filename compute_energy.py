"""Item 9 — Loihi 2 energy estimation from SLAYER SynOps.

Reads Results_lava/lava_summary.csv (produced by run_lava_infer.py),
multiplies mean synapse events per inference by Intel's published
per-event energy figure for Loihi 2, and reports:

  1. SNN-only energy on Loihi 2  (directly derived from measured SynOps)
  1b. Cross-check: an independent SynOps estimate derived from this
      codebase's own PyTorch-side spike instrumentation (B15,
      pipeline_params.json's mean_input_events_per_trial /
      mean_hidden_events_per_trial), weighted by each layer's actual
      fan-out. Reported side-by-side with (1) so a discrepancy between the
      two independent measurements is visible before either number is
      trusted, rather than only ever reporting one source.
  2. Full analog-neuromorphic pipeline energy  (SNN + cited analog front-end)
  3. Comparison to GPU / edge-CPU baselines

Usage
-----
    python compute_energy.py [--lava-dir Results_lava] [--output-dir Results_energy]

    # Also cross-check against this codebase's own measured spike counts
    # (requires training to have been run with the current code, so
    # pipeline_params.json has mean_input_events_per_trial/
    # mean_hidden_events_per_trial -- see PIPELINE_REFERENCE.md sec 10/11):
    python compute_energy.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9 \\
        --n-folds 5

Honesty note
------------
Only the Loihi 2 SNN figure (item 1) is directly measured in this work --
via Lava's own SynOps accounting for a real, currently-deployable digital
neuromorphic chip. The cross-check (1b) is a second, independent
measurement (different framework, same underlying network), included to
validate item 1, not to replace it. The analog front-end figures (items
2a-2c) are extrapolated from published silicon designs as per-stage
precedents, NOT measured in this work, and describe a separate, more
speculative hardware story: **Loihi 2 itself is a digital, asynchronous,
event-driven chip, not an analog one** -- the Gm-C/ADM/ReRAM front-end
below is a hypothetical fully-analog front-end that *could* be paired with
a digital Loihi backend, not a claim that Loihi is analog. Keep these two
stories (real digital Loihi backend vs. speculative analog front-end)
distinct when interpreting the combined "full pipeline" totals.

References
----------
[1] Davies et al., "Loihi: A neuromorphic manycore processor with on-chip
    learning," IEEE Micro, 2018.
    Published figure: 23.6 pJ/SynOp on Loihi 1 (Intel 14 nm).

[2] Orchard et al., "Efficient neuromorphic signal processing with Loihi 2,"
    IEEE SiPS, 2021.  Loihi 2 SynOp energy ~8-10 pJ (chip-level estimate).

[3] Gallegos-Ramirez et al., "A 128.7 nW Gm-C bandpass filter for EEG,"
    IEEE MWSCAS, 2014.  DOI 10.1109/MWSCAS.2014.6908555 (verified).
    128.7 nW per sub-microwatt Gm-C EEG bandpass filter -- the single
    Gm-C figure used here.

    NOTE: earlier versions of this script used "Qian et al., ISCAS 2017,
    50 nW" and "Verhoeven et al., TCAS-I 2007, 2 uW" as the Gm-C figures.
    NEITHER could be verified against a real publication -- both were
    fabricated and have been REMOVED. Do not reintroduce them.

[5] Sharifshazileh et al., "An electronic neuromorphic system for real-time
    detection of high frequency oscillations (HFO) in intracranial EEG,"
    Nature Commun., 2021.  DOI 10.1038/s41467-021-23342-2 (verified).
    Front-end signal-to-spike power: 6.2 uW/CHANNEL (whole chip 614.3 uW).
    Scaled to 22 channels: ~136 uW.  Used ONLY as the conservative
    (full analog signal-to-spike ASIC) end of the encoder term.
    NOTE: earlier versions mis-stated this as "109 uW ADM for 18 ch"; the
    real figure is 6.2 uW/ch -- corrected.

[6] Burr et al., "Neuromorphic computing using non-volatile memory,"
    Advances in Physics: X, 2017.  DOI 10.1080/23746149.2016.1259585
    (verified) -- a REVIEW of NVM crossbars for analog matrix-vector
    multiply. It does not headline one per-cell energy; the ~15 fJ/MAC used
    here is a REPRESENTATIVE in-memory-compute figure (order fJ/MAC), not a
    specific Burr number.

Energy model — SNN only
-----------------------
  E_loihi  = SynOps_per_inference x e_SynOp_J
  where e_SynOp = 23.6e-12 J  (conservative Loihi 1 [1])
              or  8.0e-12 J   (Loihi 2 estimate [2])

Cross-check model (item 1b) — this codebase's own instrumentation
--------------------------------------------------------------------
  SynOps_pytorch = input_events_per_trial * n_hidden
                   + hidden_events_per_trial * n_output
  Each layer's measured spike count is weighted by that layer's own
  fan-out (an input spike drives n_hidden downstream synapses via fc1, a
  hidden spike drives n_output synapses via fc2) -- not lumped into one
  uniform rate. Output-layer spikes are excluded: they terminate at the
  population-vote readout and drive no further weight matrix.

Energy model — whole analog pipeline (4-second MI trial at 250 Hz)
------------------------------------------------------------------
Reported as a BOUNDED RANGE, not a single number: the stages come from
independent silicon precedents (not one validated system) and the encoder
term alone spans an order of magnitude. This matches the paper's energy
section.

  Stage 1  Gm-C filter bank
           6 bands x 22 channels = 132 sub-microwatt bandpass filters
           [3]: 128.7 nW/filter x 132 = 17.0 uW  ->  x 4 s = 68 uJ

  Stage 2  CSP spatial projection (in-memory NVM crossbar)
           ~6.34 M multiply-accumulates/trial (6 bands x 22 in x 48 out
           x 1001 samples). NEGLIGIBLE: ~0.1 uJ at 15 fJ/MAC [6], and still
           only ~6 uJ even at a pessimistic 1 pJ/MAC.

  Stage 3  MIBIF comparator bank  (negligible, <1 uJ)

  Stage 4  Delta spike encoder -- DOMINANT and most implementation-dependent:
           lean digital comparator bank  ->  ~1 uJ
           full analog signal-to-spike ASIC [5]: 6.2 uW/ch x 22 x 4 s -> ~0.55 mJ

  Stage 5  SNN classifier on Loihi 2 (SynOp count x pJ/SynOp)  ->  13.9 uJ

  Whole pipeline: ~0.08 mJ (lean encoder) to ~0.63 mJ (analog encoder)/trial,
  dominated by the Gm-C filter bank and (at the high end) the encoder; CSP
  crossbar and MIBIF are negligible. Still ~95-720x below a CPU (60 mJ) and
  ~400-3000x below a GPU (250 mJ) implementation of the same pipeline.

GPU baseline (V100, inference only)
------------------------------------
  V100 TDP = 250 W; snnTorch batch=1 forward pass ~1 ms.
  E_GPU_full = 250 W x 1e-3 s = 250 mJ.
  E_GPU_30pct = 75 W x 1e-3 s = 75 mJ  (30% utilisation, more realistic).

Edge-CPU baseline (ARM Cortex-A72, ~3 W)
-----------------------------------------
  FBCSP + SNN on CPU: digital filter + CSP + encode + SNN ~20 ms.
  E_CPU = 3 W x 20e-3 s = 60 mJ.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
# Analog front-end energy constants (4-second MI trial)
# Per-stage figures from DOI-verified published silicon — NOT measured here.
# A hypothetical all-analog front-end that COULD be paired with a digital
# Loihi backend; Loihi 2 itself is digital (see module docstring). The Qian
# and Verhoeven Gm-C figures used in earlier versions could not be verified
# and have been removed; the Gm-C term now uses Gallegos 2014 (DOI-verified).
# ---------------------------------------------------------------------------

TRIAL_DURATION_S   = 4.0   # seconds (T=1001 at 250 Hz ≈ 4 s)
N_EEG_CHANNELS     = 22

# Gm-C filter bank: one sub-microwatt Gm-C EEG bandpass filter per
# (band, channel) → 6 bands × 22 channels = 132 filters.
N_GMC_FILTERS      = 6 * 22                                    # = 132
E_GMC_J            = 128.7e-9 * N_GMC_FILTERS * TRIAL_DURATION_S   # [3]  ≈68 µJ

# CSP spatial projection on an in-memory NVM crossbar [6].
# ~6.34 M multiply-accumulates/trial: 6 bands × 22 in × (6 pairs × 2m=8) out
# × 1001 samples. Per-MAC energy is a REPRESENTATIVE order-fJ figure, not a
# specific Burr number (Burr 2017 is a review).
N_SAMPLES          = 1001
N_CSP_MACS         = N_SAMPLES * 6 * N_EEG_CHANNELS * (6 * 8)  # ≈6.34 M
E_CROSSBAR_PER_MAC = 15e-15                                    # representative fJ/MAC
E_CSP_J            = N_CSP_MACS * E_CROSSBAR_PER_MAC           # ≈0.1 µJ (negligible)

# MIBIF comparator bank — negligible
E_MIBIF_J          = 0.5e-6                                    # <1 µJ

# Delta spike encoder — DOMINANT and most implementation-dependent term.
#   (a) lean digital comparator bank over 288 features × 1001 samples
E_ENCODER_DIGITAL_J = 1e-6                                     # ~1 µJ
#   (b) full analog signal-to-spike front end, Sharifshazileh 2021 [5]:
#       6.2 µW/channel (verified; whole chip 614.3 µW), scaled to 22 ch.
E_ENCODER_ANALOG_J  = 6.2e-6 * N_EEG_CHANNELS * TRIAL_DURATION_S   # ≈0.55 mJ


# ---------------------------------------------------------------------------
# Load Lava summary CSV
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
# Cross-check: this codebase's own PyTorch-side spike instrumentation (B15)
# ---------------------------------------------------------------------------

def load_pytorch_side_synops(
    results_dir: Path,
    subjects: List[int],
    n_folds: int,
) -> Dict[int, float]:
    """Compute an independent, fan-out-weighted SynOps estimate per subject
    from this codebase's own PyTorch-side spike counts (B15), averaged
    across folds.

    Uses ``mean_input_events_per_trial`` / ``mean_hidden_events_per_trial``
    from each fold's ``pipeline_params.json`` -- these are measured
    directly via ``evaluate_model_with_event_breakdown`` during training,
    not assumed. Requires training to have been run with the current code
    (folds trained before this instrumentation existed are silently
    skipped with a logged warning, not silently zero-filled).

    Parameters
    ----------
    results_dir : Path
        Root results directory (e.g. ``Results``).
    subjects : List[int]
        Subject IDs to load.
    n_folds : int
        Number of folds per subject.

    Returns
    -------
    Dict[int, float]
        ``subject_id -> mean SynOps per trial across that subject's folds``.
        Subjects with no usable folds are omitted.
    """
    per_subject: Dict[int, float] = {}
    for subject_id in subjects:
        fold_synops: List[float] = []
        for fold in range(n_folds):
            path = results_dir / f"Subject_{subject_id}" / f"fold_{fold}" / "pipeline_params.json"
            if not path.exists():
                continue
            with open(path) as f:
                params = json.load(f)
            if "mean_input_events_per_trial" not in params or "mean_hidden_events_per_trial" not in params:
                logger.warning(
                    "Subject %d fold %d predates the per-layer event "
                    "breakdown (B15) -- skipped from cross-check.",
                    subject_id, fold,
                )
                continue
            n_hidden = params["hidden_neurons"]
            n_output = params["n_classes"] * params["population_per_class"]
            synops = (
                params["mean_input_events_per_trial"] * n_hidden
                + params["mean_hidden_events_per_trial"] * n_output
            )
            fold_synops.append(synops)
        if fold_synops:
            per_subject[subject_id] = float(np.mean(fold_synops))
    return per_subject


def _print_cross_check(lava_rows: List[Dict], pytorch_synops: Dict[int, float]) -> None:
    """Print Lava-measured vs. PyTorch-side SynOps, side by side, per subject.

    A large disagreement here means the two independent measurements of
    "how much does this network actually spike" don't agree -- worth
    investigating before trusting either number in the paper.
    """
    if not pytorch_synops:
        print("\n(No PyTorch-side cross-check data available -- pass "
              "--results-dir/--subjects/--n-folds pointing at folds trained "
              "with the current code to enable this.)\n")
        return

    print(f"\n{'='*72}")
    print(f"  Cross-check: Lava-measured vs. PyTorch-side SynOps estimate")
    print(f"  (independent measurements of the same underlying network --")
    print(f"   large disagreement means something needs investigating)")
    print(f"{'='*72}")
    print(f"  {'Subj':<6}  {'Lava SynOps':>14}  {'PyTorch SynOps':>16}  {'Ratio':>8}")
    print("  " + "-" * 52)

    ratios: List[float] = []
    for r in lava_rows:
        subject_id = r["subject"]
        lava_synops = r["synops_mean"]
        pt_synops = pytorch_synops.get(subject_id)
        if pt_synops is None:
            print(f"  S{subject_id:<5}  {lava_synops:>14,.0f}  {'(no data)':>16}  {'--':>8}")
            continue
        ratio = pt_synops / lava_synops if lava_synops > 0 else float("nan")
        ratios.append(ratio)
        print(f"  S{subject_id:<5}  {lava_synops:>14,.0f}  {pt_synops:>16,.0f}  {ratio:>8.3f}")

    if ratios:
        print("  " + "-" * 52)
        print(f"  Mean ratio (PyTorch / Lava): {np.mean(ratios):.3f}  "
              f"(1.0 = perfect agreement)")
    print(f"{'='*72}\n")


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

def _print_frontend_breakdown(mean_snn_l2_uj: float, mean_snn_l1_uj: float) -> None:
    """Print the whole-pipeline energy budget as a BOUNDED RANGE.

    The range is set by the encoder: a lean digital comparator bank
    (~1 uJ) vs. a full analog signal-to-spike ASIC (~0.55 mJ). Per-stage
    figures are DOI-verified precedents from INDEPENDENT designs, not one
    validated system -- hence a range, not a single number. Matches the
    paper's energy section.
    """
    e_gmc_uj      = E_GMC_J             * 1e6
    e_csp_uj      = E_CSP_J             * 1e6
    e_mibif_uj    = E_MIBIF_J           * 1e6
    e_enc_lean_uj = E_ENCODER_DIGITAL_J * 1e6
    e_enc_con_uj  = E_ENCODER_ANALOG_J  * 1e6

    fixed_uj      = e_gmc_uj + e_csp_uj + e_mibif_uj + mean_snn_l2_uj
    total_lean_uj = fixed_uj + e_enc_lean_uj
    total_con_uj  = fixed_uj + e_enc_con_uj

    e_cpu_uj      = E_CPU_J          * 1e6
    e_gpu_full_uj = E_GPU_FULL_TDP_J * 1e6

    print(f"\n{'='*78}")
    print(f"  Whole-Pipeline Energy (4-second MI trial) -- BOUNDED RANGE")
    print(f"  The SNN stage is a spike-count estimate for a real digital chip")
    print(f"  (Loihi 2). The analog front-end stages are DOI-verified per-stage")
    print(f"  precedents from INDEPENDENT designs, not one validated system --")
    print(f"  reported as a range, not a single figure.")
    print(f"{'='*78}")
    print(f"  {'Stage':<36}  {'Energy (uJ)':>15}  Reference")
    print(f"  {'-'*74}")
    stages = [
        ("Gm-C filter bank (132 filters)",  f"{e_gmc_uj:7.1f}",                       "Gallegos 2014 [3]"),
        ("CSP crossbar (~6.3M MAC)",        f"{e_csp_uj:7.1f}",                       "Burr 2017 [6] (repr.)"),
        ("MIBIF comparator bank",           f"{e_mibif_uj:7.1f}",                     "negligible"),
        ("Spike encoder (lean .. analog)",  f"{e_enc_lean_uj:.0f} .. {e_enc_con_uj:.0f}", "digital .. Sharif 2021 [5]"),
        ("SNN classifier (Loihi 2)",        f"{mean_snn_l2_uj:7.1f}",                 "This work"),
    ]
    for name, e, ref in stages:
        print(f"  {name:<36}  {e:>15}  {ref}")
    print(f"  {'-'*74}")
    print(f"  {'WHOLE PIPELINE':<36}  {total_lean_uj:6.0f} .. {total_con_uj:<6.0f} uJ"
          f"  (~{total_lean_uj/1000:.2f}-{total_con_uj/1000:.2f} mJ)")
    print(f"{'='*78}")

    print(f"\n  vs a digital implementation of the SAME pipeline:")
    print(f"    Edge CPU (ARM A72, 3 W, 20 ms):  {e_cpu_uj/1000:.0f} mJ"
          f"   ->  {e_cpu_uj/total_con_uj:.0f}-{e_cpu_uj/total_lean_uj:.0f}x more efficient")
    print(f"    GPU V100 (full TDP, 1 ms):       {e_gpu_full_uj/1000:.0f} mJ"
          f"   ->  {e_gpu_full_uj/total_con_uj:.0f}-{e_gpu_full_uj/total_lean_uj:.0f}x more efficient")
    print(f"\n  Front-end-dominated: the SNN classifier ({mean_snn_l2_uj:.1f} uJ) is a small")
    print(f"  slice; the always-on analog front end sets the budget. The headline")
    print(f"  classifier-only margin ({e_cpu_uj/mean_snn_l2_uj:,.0f}x vs CPU) narrows to")
    print(f"  ~1e2x once the front end is counted.\n")


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
    p.add_argument("--results-dir", default=None,
                   help="If set, also cross-check Lava's measured SynOps "
                        "against this codebase's own PyTorch-side spike "
                        "counts (B15) from this results directory (e.g. "
                        "'Results'). Requires training to have been run "
                        "with the current code.")
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="Subject IDs for the cross-check (required if "
                        "--results-dir is set).")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Folds per subject for the cross-check.")
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

    if args.results_dir is not None:
        if args.subjects is None:
            logger.error("--subjects is required when --results-dir is set")
        else:
            pytorch_synops = load_pytorch_side_synops(
                Path(args.results_dir), args.subjects, args.n_folds
            )
            _print_cross_check(rows, pytorch_synops)

    mean_l2_uj = float(np.mean([r["energy_loihi2_uJ"] for r in rows]))
    mean_l1_uj = float(np.mean([r["energy_loihi1_uJ"] for r in rows]))
    _print_frontend_breakdown(mean_l2_uj, mean_l1_uj)

    _write_csv(rows, output_dir)


if __name__ == "__main__":
    main()
