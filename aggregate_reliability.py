"""Aggregate reliability_results.json across all subjects/folds for the paper figure.

Two-level aggregation, matching Table III's convention: average each
subject's 5 folds first, then report mean +/- SD across the 9 subjects
(cross-subject SD, not within-fold Monte Carlo SD).

Usage (on Puhti, from the project root):
    python aggregate_reliability.py --results-dir Results_verify
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SWEEPS = [
    "csp_weight_noise", "snn_weight_noise", "beta_noise",
    "filter_bank_noise", "ea_whitener_noise", "znorm_noise",
    "encoder_threshold_noise", "encoder_adaptation_noise",
    "joint_noise_all_sources",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="Results_verify")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    severities = None
    clean_per_subject: list[float] = []
    sweep_per_subject: dict[str, dict[str, list[float]]] = {s: {} for s in SWEEPS}
    missing: list[str] = []

    for subj in args.subjects:
        fold_clean: list[float] = []
        fold_vals: dict[str, dict[str, list[float]]] = {s: {} for s in SWEEPS}
        for fold in range(args.n_folds):
            p = results_dir / f"Subject_{subj}" / f"fold_{fold}" / "reliability_results.json"
            if not p.exists():
                missing.append(str(p))
                continue
            with open(p) as fh:
                d = json.load(fh)
            fold_clean.append(d["clean"]["acc"])
            if severities is None:
                severities = d["severities"]
            for sweep in SWEEPS:
                for sev in severities:
                    key = str(sev)
                    fold_vals[sweep].setdefault(key, []).append(d[sweep][key]["acc_mean"])

        if not fold_clean:
            print(f"Subject {subj}: NO DATA")
            continue
        clean_per_subject.append(float(np.mean(fold_clean)))
        for sweep in SWEEPS:
            for sev in severities:
                key = str(sev)
                subj_mean = float(np.mean(fold_vals[sweep][key]))
                sweep_per_subject[sweep].setdefault(key, []).append(subj_mean)

    if missing:
        print(f"WARNING: {len(missing)} missing fold files:")
        for m in missing:
            print(f"  {m}")
        print()

    print(f"clean_acc,mean={np.mean(clean_per_subject):.6f},std={np.std(clean_per_subject):.6f},n={len(clean_per_subject)}")
    print()
    print("sweep,severity,acc_mean,acc_std,n_subjects")
    for sweep in SWEEPS:
        for sev in severities:
            vals = np.array(sweep_per_subject[sweep][str(sev)])
            print(f"{sweep},{sev},{vals.mean():.6f},{vals.std():.6f},{len(vals)}")


if __name__ == "__main__":
    main()
