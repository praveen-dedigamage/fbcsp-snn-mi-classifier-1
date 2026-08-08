#!/usr/bin/env python
"""Diagnose why held-out accuracy is far above the previously reported value.

Context: BNCI2015-001 folds trained on Roihu report 92-99% on the held-out
session, where the same pipeline previously reported 72.1% mean and its own
FBCSP+LDA baseline reported 76.7%.  Either the pipeline genuinely improved, or
information is reaching the test set.  This script tries to tell those apart
using cheap checks that need no retraining.

It answers four questions:

1. Are the train and test sessions actually disjoint recordings?  If the
   loader ever returned the same session twice, or the sessions share epochs,
   test accuracy would approach 100% with no bug anywhere else.
2. Is the test set balanced, so 50% really is chance?
3. Do the classical baselines move together with the SNN?  LDA and SVM are
   fitted on the same front end and scored on the same held-out session.  If
   they are also far above their published values, the cause is upstream of
   the classifier and affects everything equally.  If only the SNN moved, the
   cause is in the spiking path.
4. What configuration produced these artifacts?

Usage
-----
::

    python diagnose_leakage.py --subject 1 --results-dir Results_bnci2015

A decisive follow-up that this script does *not* run, because it costs a full
training job: permute the training labels and retrain.  With labels destroyed,
held-out accuracy must fall to chance.  If it does not, information is leaking
regardless of what any of the checks below say.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from fbcsp_snn import setup_logger

    logger = setup_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)


def _flat_normalised(X: np.ndarray) -> np.ndarray:
    """Flatten each trial and scale to unit norm, for cosine comparison.

    Parameters
    ----------
    X : np.ndarray
        Array of shape ``(n_trials, n_channels, n_samples)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_channels * n_samples)``, each row unit norm.
    """
    F = X.reshape(len(X), -1).astype(np.float64)
    F -= F.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return F / norms


def check_disjoint(
    X_train: np.ndarray, X_test: np.ndarray, threshold: float = 0.999
) -> Dict[str, Any]:
    """Look for test trials that duplicate a training trial.

    Uses cosine similarity between flattened trials.  Genuine cross-session
    recordings of the same subject correlate weakly; anything at ~1.0 is the
    same epoch appearing twice.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Trial arrays ``(n_trials, n_channels, n_samples)``.
    threshold : float
        Similarity at or above which two trials are called duplicates.

    Returns
    -------
    dict
        Summary including the duplicate count and the similarity distribution.
    """
    A = _flat_normalised(X_train)
    B = _flat_normalised(X_test)
    sim = B @ A.T  # (n_test, n_train) cosine similarity

    best = sim.max(axis=1)
    n_dup = int((best >= threshold).sum())

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "identical_arrays": bool(
            X_train.shape == X_test.shape and np.array_equal(X_train, X_test)
        ),
        "duplicate_test_trials": n_dup,
        "max_similarity": round(float(best.max()), 6),
        "mean_best_similarity": round(float(best.mean()), 6),
        "median_best_similarity": round(float(np.median(best)), 6),
        "threshold": threshold,
    }


def check_balance(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Report class counts so the chance level is explicit."""
    def counts(y: np.ndarray) -> Dict[str, int]:
        u, c = np.unique(y, return_counts=True)
        return {str(int(k)): int(v) for k, v in zip(u, c)}

    ct = counts(y_test)
    majority = max(ct.values()) / sum(ct.values()) if ct else float("nan")
    return {
        "train_counts": counts(y_train),
        "test_counts": ct,
        "test_majority_class_rate": round(majority * 100, 2),
    }


def read_folds(results_dir: Path, subject: int) -> List[Dict[str, Any]]:
    """Load every fold's ``pipeline_params.json`` for one subject."""
    out: List[Dict[str, Any]] = []
    sub = results_dir / f"Subject_{subject}"
    if not sub.is_dir():
        logger.warning("no such directory: %s", sub)
        return out
    for fold_dir in sorted(sub.glob("fold_*"), key=lambda p: int(p.name.split("_")[1])):
        path = fold_dir / "pipeline_params.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                out.append({"fold": int(fold_dir.name.split("_")[1]), **json.load(f)})
    return out


def compare_classifiers(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare SNN against LDA/SVM on the same held-out session.

    All three are scored on identical data through an identical front end, so
    a gap between them localises the cause.
    """
    def mean_pct(key: str) -> Optional[float]:
        vals = [f[key] for f in folds if f.get(key) is not None]
        return round(float(np.mean(vals)) * 100, 2) if vals else None

    snn, lda, svm = (
        mean_pct("test_acc_fp32"),
        mean_pct("test_acc_lda"),
        mean_pct("test_acc_svm"),
    )
    return {
        "snn_test_acc": snn,
        "lda_test_acc": lda,
        "svm_test_acc": svm,
        "snn_val_acc": mean_pct("val_acc_fp32"),
        "paper_reference": {"snn": 72.1, "lda": 76.7, "svm": 72.0},
        "interpretation": (
            "LDA/SVM also far above their paper values -> cause is upstream of "
            "the classifier (data, front end, or evaluation), affecting all "
            "methods equally. Only the SNN elevated -> cause is in the spiking "
            "path (encoding, MIBIF, or training)."
        ),
    }


def main() -> None:
    """Run the cheap diagnostics for one subject and print a verdict."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--dataset", default="BNCI2015_001")
    ap.add_argument("--results-dir", default="Results_bnci2015")
    ap.add_argument("--output", default=None, help="write findings as JSON")
    args = ap.parse_args()

    from fbcsp_snn.config import Config
    from fbcsp_snn.pipeline import _load_raw

    cfg = Config()
    cfg.source = "moabb"
    cfg.moabb_dataset = args.dataset
    cfg.subject_id = args.subject
    cfg.n_classes = 2

    X_train, y_train, X_test, y_test = _load_raw(cfg)

    findings: Dict[str, Any] = {
        "dataset": args.dataset,
        "subject": args.subject,
        "disjointness": check_disjoint(X_train, X_test),
        "balance": check_balance(y_train, y_test),
    }

    folds = read_folds(Path(args.results_dir), args.subject)
    findings["n_folds_found"] = len(folds)
    if folds:
        findings["classifiers"] = compare_classifiers(folds)
        keep = (
            "dataset", "n_classes", "seed", "use_amp", "bands", "filter_type",
            "encoder_type", "euclidean_alignment", "riemannian_mean",
            "csp_dual_end", "csp_m", "mi_fraction", "feature_method",
            "n_features_selected", "n_timesteps", "hidden_neurons",
            "population_per_class", "loss_type",
        )
        findings["config"] = {k: folds[0].get(k) for k in keep}

    logger.info("=" * 68)
    d = findings["disjointness"]
    logger.info("train %d trials | test %d trials", d["n_train"], d["n_test"])
    logger.info("train and test arrays identical : %s", d["identical_arrays"])
    logger.info("test trials duplicating a train trial (cos >= %.3f) : %d",
                d["threshold"], d["duplicate_test_trials"])
    logger.info("similarity to nearest train trial: max %.4f  median %.4f",
                d["max_similarity"], d["median_best_similarity"])

    b = findings["balance"]
    logger.info("test class counts %s -> majority baseline %.1f%%",
                b["test_counts"], b["test_majority_class_rate"])

    if "classifiers" in findings:
        c = findings["classifiers"]
        logger.info("-" * 68)
        logger.info("held-out accuracy over %d folds", len(folds))
        logger.info("  SNN %.2f%%   (paper 72.1%%)", c["snn_test_acc"])
        logger.info("  LDA %.2f%%   (paper 76.7%%)", c["lda_test_acc"])
        logger.info("  SVM %.2f%%   (paper 72.0%%)", c["svm_test_acc"])
        logger.info("-" * 68)
        gap_snn = c["snn_test_acc"] - 72.1
        gap_lda = c["lda_test_acc"] - 76.7
        if gap_lda > 8.0:
            logger.info("VERDICT: LDA moved by %+.1f pts too. The cause is "
                        "upstream of the classifier.", gap_lda)
        elif gap_snn > 8.0:
            logger.info("VERDICT: SNN moved %+.1f pts while LDA moved %+.1f. "
                        "The cause is in the spiking path.", gap_snn, gap_lda)
        else:
            logger.info("VERDICT: both close to published values.")
    logger.info("=" * 68)
    logger.info("Not covered here: a label-permutation control. Retrain one "
                "fold with shuffled training labels; held-out accuracy must "
                "collapse to chance. Any leak survives label permutation.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(findings, f, indent=2)
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
