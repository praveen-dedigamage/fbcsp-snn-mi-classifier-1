"""Train the non-spiking ANN twin on one already-fitted fold, reusing the
SAME saved CSP filters, z-norm stats, and MIBIF selector as the SNN --
i.e. the exact same spike-encoded input sequence -- to isolate whether
the LIF spiking mechanism itself contributes anything, holding
architecture, optimiser, training protocol, and input features fixed.

Two loss variants share this same script (ANNClassifier + train_fold
already support both without any training-loop changes):
  --loss-type van_rossum      (max parity with the SNN's own loss)
  --loss-type cross_entropy   (the "conventional" loss a plain ANN would
                                normally use -- already implemented as an
                                existing ablation option, just applied to
                                a non-spiking model here for the first time)

Usage:
    python run_ann_twin.py --results-dir Results_verify \\
        --subject-id 1 --fold 0 --moabb-dataset BNCI2014_001 \\
        --loss-type van_rossum
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from fbcsp_snn import DEVICE
from fbcsp_snn.datasets import DATASET_REGISTRY, load_moabb
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import ANNClassifier
from fbcsp_snn.training import evaluate_model, train_fold


def _concat_projections(proj: dict) -> np.ndarray:
    """Matches fbcsp_snn/pipeline.py's _concat_projections exactly."""
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True,
                     help="Existing results dir from a COMPLETED SNN "
                          "training run (reuses its saved csp_filters.pkl, "
                          "znorm.pkl, mibif.pkl, pipeline_params.json).")
    ap.add_argument("--subject-id", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--moabb-dataset", required=True)
    ap.add_argument("--n-folds", type=int, default=5,
                     help="Every real run uses n_folds=5 with plain "
                          "StratifiedKFold -- matches that unconditionally.")
    ap.add_argument("--loss-type", choices=["van_rossum", "cross_entropy"],
                     default="van_rossum")
    ap.add_argument("--base-thresh", type=float, default=0.001)
    ap.add_argument("--adapt-inc", type=float, default=0.6)
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    fold_dir = os.path.join(args.results_dir, f"Subject_{args.subject_id}",
                             f"fold_{args.fold}")
    print(f"Loading saved fold artifacts from {fold_dir} ...")
    with open(os.path.join(fold_dir, "pipeline_params.json")) as f:
        params = json.load(f)
    with open(os.path.join(fold_dir, "csp_filters.pkl"), "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(os.path.join(fold_dir, "znorm.pkl"), "rb") as f:
        znorm: ZNormaliser = pickle.load(f)
    mibif: MIBIFSelector | None = None
    mibif_path = os.path.join(fold_dir, "mibif.pkl")
    if os.path.exists(mibif_path):
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    bands = [tuple(b) for b in params["bands"]]
    sfreq = float(DATASET_REGISTRY[args.moabb_dataset]["sfreq"])
    encoder_type = params.get("encoder_type", "delta")
    n_classes = params["n_classes"]
    hidden_neurons = params.get("hidden_neurons", 64)
    population_per_class = params.get("population_per_class", 20)
    beta = params.get("beta", 0.95)
    tau_vr = params.get("tau_vr", 10.0)
    print(f"  bands={bands}  sfreq={sfreq}  encoder={encoder_type}  "
          f"loss_type={args.loss_type}")

    # The preprocessing chain below (raw load -> filter bank -> saved CSP ->
    # saved z-norm -> encode -> saved MIBIF) is IDENTICAL regardless of
    # --loss-type -- only the model training that follows differs. Cache it
    # once per fold so a second --loss-type run (e.g. cross_entropy after
    # van_rossum already ran) skips straight to training instead of
    # redoing several minutes of identical CPU-side work for nothing.
    cache_path = os.path.join(fold_dir, "ann_twin_inputs.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached preprocessed inputs from {cache_path} "
              f"(shared across --loss-type runs, computed once) ...")
        cache = torch.load(cache_path, weights_only=False)
        spikes_tr, spikes_val, spikes_te = (
            cache["spikes_tr"], cache["spikes_val"], cache["spikes_te"]
        )
        y_f_tr_0, y_f_val_0, y_test_0 = (
            cache["y_f_tr_0"], cache["y_f_val_0"], cache["y_test_0"]
        )
    else:
        print(f"No cache at {cache_path} -- computing (will be cached for "
              f"any other --loss-type run of this same fold) ...")
        print(f"Loading raw MOABB data: {args.moabb_dataset}, "
              f"subject {args.subject_id} ...")
        X_train, y_train, X_test, y_test = load_moabb(
            args.moabb_dataset, args.subject_id
        )

        # Reproduce the EXACT train/val split the SNN's own training run used.
        splitter = StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=42
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(
            splitter.split(X_train, y_train)
        ):
            if fold_idx == args.fold:
                break
        else:
            raise ValueError(f"Fold {args.fold} not found (n_folds={args.n_folds})")

        X_f_tr,  y_f_tr  = X_train[tr_idx],  y_train[tr_idx]
        X_f_val, y_f_val = X_train[val_idx], y_train[val_idx]

        print("Applying filter bank + SAVED (fitted) CSP + SAVED (fitted) "
              "z-norm to train/val/test ...")
        Xb_tr  = apply_filter_bank(X_f_tr,  bands, sfreq)
        Xb_val = apply_filter_bank(X_f_val, bands, sfreq)
        Xb_te  = apply_filter_bank(X_test,  bands, sfreq)

        proj_tr  = csp.transform(Xb_tr)
        proj_val = csp.transform(Xb_val)
        proj_te  = csp.transform(Xb_te)

        X_concat_tr  = _concat_projections(proj_tr).astype(np.float32)
        X_concat_val = _concat_projections(proj_val).astype(np.float32)
        X_concat_te  = _concat_projections(proj_te).astype(np.float32)
        del Xb_tr, Xb_val, Xb_te, proj_tr, proj_val, proj_te

        X_norm_tr  = znorm.transform(X_concat_tr)
        X_norm_val = znorm.transform(X_concat_val)
        X_norm_te  = znorm.transform(X_concat_te)
        del X_concat_tr, X_concat_val, X_concat_te

        print(f"Encoding spikes (base_thresh={args.base_thresh}, "
              f"adapt_inc={args.adapt_inc}, decay={args.decay}) ...")
        spikes_tr  = encode_tensor(
            torch.from_numpy(X_norm_tr).permute(2, 0, 1),
            args.base_thresh, args.adapt_inc, args.decay, encoder_type,
        )
        spikes_val = encode_tensor(
            torch.from_numpy(X_norm_val).permute(2, 0, 1),
            args.base_thresh, args.adapt_inc, args.decay, encoder_type,
        )
        spikes_te  = encode_tensor(
            torch.from_numpy(X_norm_te).permute(2, 0, 1),
            args.base_thresh, args.adapt_inc, args.decay, encoder_type,
        )
        del X_norm_tr, X_norm_val, X_norm_te

        if mibif is not None:
            spikes_tr  = mibif.transform(spikes_tr)
            spikes_val = mibif.transform(spikes_val)
            spikes_te  = mibif.transform(spikes_te)

        y_f_tr_0  = y_f_tr  - 1
        y_f_val_0 = y_f_val - 1
        y_test_0  = y_test  - 1

        torch.save({
            "spikes_tr": spikes_tr, "spikes_val": spikes_val, "spikes_te": spikes_te,
            "y_f_tr_0": y_f_tr_0, "y_f_val_0": y_f_val_0, "y_test_0": y_test_0,
        }, cache_path)
        print(f"  Cached preprocessed inputs to {cache_path}")

    n_input = spikes_tr.shape[2]
    print(f"  n_input={n_input}  T={spikes_tr.shape[0]}  "
          f"(SAME spike-encoded input the SNN received for this fold)")

    print(f"Training ANNClassifier (non-spiking twin, loss={args.loss_type}) ...")
    model = ANNClassifier(
        n_input=n_input, n_hidden=hidden_neurons, n_classes=n_classes,
        population_per_class=population_per_class, beta=beta,
        dropout_prob=0.5,
    )
    # Separate subdirectory so this never overwrites the SNN's own
    # best_model.pt in the same fold_dir.
    ann_fold_dir = Path(fold_dir) / f"ann_twin_{args.loss_type}"

    result = train_fold(
        spikes_train=spikes_tr,
        y_train=y_f_tr_0,
        spikes_val=spikes_val,
        y_val=y_f_val_0,
        model=model,
        n_classes=n_classes,
        population_per_class=population_per_class,
        tau_vr=tau_vr,
        loss_type=args.loss_type,
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
        device=DEVICE,
        fold_dir=ann_fold_dir,
    )

    test_acc, _ = evaluate_model(model, spikes_te, y_test_0, DEVICE)
    print(f"  best_val_acc={result.best_val_acc:.4f}  "
          f"best_epoch={result.best_epoch}  "
          f"stopped_epoch={result.stopped_epoch}  test_acc={test_acc:.4f}")

    out = {
        "loss_type": args.loss_type,
        "val_acc": result.best_val_acc,
        "test_acc": test_acc,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
    }
    out_path = os.path.join(fold_dir, f"ann_twin_{args.loss_type}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
