"""Pre-compute and save test spike tensors for each fold.

Runs in the MAIN venv (.venv) which has moabb, snnTorch, and all pipeline
dependencies.  Saves test_spikes.pt and test_labels.npy into each fold
directory so that run_lava_infer.py (lava venv) can load them without
needing moabb / MNE.

Usage
-----
    python save_test_spikes.py \\
        --results-dir Results_adm_static6_ptq \\
        --subjects 1 2 3 4 5 6 7 8 9 \\
        --n-folds 5 \\
        --moabb-dataset BNCI2014_001
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).parent))

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.datasets import load_moabb
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser

logger: logging.Logger = setup_logger(__name__)


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
        out.append(filtered.reshape(n_trials, n_channels, n_samples).astype(np.float32))
    return out


def save_subject_spikes(
    subject_id: int,
    results_dir: Path,
    n_folds: int,
    X_test: np.ndarray,
    y_test_0: np.ndarray,
    sfreq: float,
) -> None:
    subject_dir = results_dir / f"Subject_{subject_id}"

    for fold_idx in range(n_folds):
        fold_dir = subject_dir / f"fold_{fold_idx}"
        out_spikes = fold_dir / "test_spikes.pt"
        out_labels = fold_dir / "test_labels.npy"

        if out_spikes.exists() and out_labels.exists():
            logger.info("S%d fold %d: already saved — skipping", subject_id, fold_idx)
            continue

        if not fold_dir.exists():
            logger.warning("S%d fold %d: directory missing — skipping", subject_id, fold_idx)
            continue

        params_path = fold_dir / "pipeline_params.json"
        if not params_path.exists():
            logger.warning("S%d fold %d: pipeline_params.json missing", subject_id, fold_idx)
            continue

        with open(params_path) as f:
            params = json.load(f)

        try:
            with open(fold_dir / "csp_filters.pkl", "rb") as f:
                csp: PairwiseCSP = pickle.load(f)
            with open(fold_dir / "znorm.pkl", "rb") as f:
                znorm: ZNormaliser = pickle.load(f)
        except FileNotFoundError as exc:
            logger.warning("S%d fold %d: artifact missing (%s)", subject_id, fold_idx, exc)
            continue

        mibif: Optional[MIBIFSelector] = None
        mibif_path = fold_dir / "mibif.pkl"
        if mibif_path.exists():
            with open(mibif_path, "rb") as f:
                mibif = pickle.load(f)

        bands: List[Tuple[float, float]] = [tuple(b) for b in params["bands"]]
        encoder_type = params.get("encoder_type", "delta")
        base_thresh   = params.get("base_thresh", 0.001)
        adapt_inc     = params.get("adapt_inc", 0.6)
        decay         = params.get("decay", 0.95)

        # Apply nominal (clean) filter bank
        sos_clean  = [_band_sos(lo, hi, sfreq) for lo, hi in bands]
        X_bands    = _apply_filterbank(X_test, sos_clean)
        proj       = csp.transform(X_bands)

        # Concatenate projections and z-normalise
        X_concat = np.concatenate(
            [proj[p] for p in sorted(proj.keys())], axis=1
        ).astype(np.float32)                          # (n_trials, n_feat, n_samples)
        X_norm   = znorm.transform(X_concat)
        X_t      = torch.from_numpy(X_norm).to(DEVICE).permute(2, 0, 1)  # (T, B, F)

        # Spike encoding
        spikes = encode_tensor(X_t, base_thresh, adapt_inc, decay, encoder_type)
        if mibif is not None:
            spikes = mibif.transform(spikes)

        # Save: spikes (T, n_trials, n_features) on CPU, labels as numpy
        torch.save(spikes.cpu(), out_spikes)
        np.save(out_labels, y_test_0)

        logger.info(
            "S%d fold %d: spikes %s → %s",
            subject_id, fold_idx, str(spikes.shape), out_spikes,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Save test spike tensors for Lava simulation.")
    ap.add_argument("--results-dir",   default="Results_adm_static6_ptq")
    ap.add_argument("--subjects",      nargs="+", type=int, default=list(range(1, 10)))
    ap.add_argument("--n-folds",       type=int, default=5)
    ap.add_argument("--moabb-dataset", default="BNCI2014_001")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    sfreq = 250.0

    for subject_id in args.subjects:
        logger.info("Subject %d", subject_id)
        _, _, X_test, y_test = load_moabb(args.moabb_dataset, subject_id)
        y_test_0 = y_test - 1
        save_subject_spikes(subject_id, results_dir, args.n_folds,
                            X_test, y_test_0, sfreq)

    logger.info("Done — test spikes saved to fold directories.")


if __name__ == "__main__":
    main()
