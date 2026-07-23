"""Aggregate ann_twin_<loss_type>_results.json across folds/subjects into
a per-subject mean +/- SD table, matching the convention of the existing
baseline tables in the paper.

Usage:
    python aggregate_ann_twin.py --results-dir Results_verify \\
        --loss-type van_rossum --subjects 1 2 3 4 5 6 7 8 9 --n-folds 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--loss-type", choices=["van_rossum", "cross_entropy"],
                     required=True)
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    print("-" * 50)
    print(f"{'Subj':<6}{'ANN twin (' + args.loss_type + ')':<30}{'Folds':<6}")
    print("-" * 50)

    per_subject: dict[int, float] = {}
    missing: list[str] = []

    for subj in args.subjects:
        vals = []
        for fold in range(args.n_folds):
            path = os.path.join(
                args.results_dir, f"Subject_{subj}", f"fold_{fold}",
                f"ann_twin_{args.loss_type}_results.json",
            )
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                r = json.load(f)
            vals.append(r["test_acc"] * 100)

        if not vals:
            print(f"S{subj:<5}{'(no data)':<30}{0:<6}")
            continue

        mean, sd = np.mean(vals), np.std(vals, ddof=0)
        per_subject[subj] = mean
        print(f"S{subj:<5}{f'{mean:.1f}+/-{sd:.1f}':<30}{len(vals):<6}")

    print("-" * 50)
    if per_subject:
        all_vals = np.array(list(per_subject.values()))
        print(f"{'Mean':<6}{f'{all_vals.mean():.1f}+/-{all_vals.std(ddof=0):.1f}':<30}"
              f"{len(per_subject) * args.n_folds:<6}")
    print("-" * 50)

    if missing:
        print(f"\n{len(missing)} fold(s) missing "
              f"ann_twin_{args.loss_type}_results.json:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
