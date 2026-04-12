"""Top-level pipeline orchestrators: run_train, run_infer, run_aggregate.

Each function consumes a :class:`~fbcsp_snn.config.Config` object and
runs the corresponding pipeline phase end-to-end, persisting all artifacts
under ``Results/Subject_<N>/fold_<K>/``.

Artifact layout per fold
------------------------
::

    Results/Subject_N/
      fold_K/
        best_model.pt           — PyTorch state dict (FP32 best val)
        csp_filters.pkl         — PairwiseCSP instance (pickle)
        znorm.pkl               — ZNormaliser instance (pickle)
        mibif.pkl               — MIBIFSelector instance (pickle)
        pipeline_params.json    — bands, metrics, hyperparams (per fold)
        band_selection.png      — Fisher curve + selected bands
        spike_propagation.png   — spike raster for 4 training trials
        neuron_traces.png       — output LIF membrane + spike overlay
        weight_histograms.png   — FP32 vs INT8-sim weight distributions
        confusion_fp32.png      — normalised confusion matrix (FP32)
        confusion_int8.png      — normalised confusion matrix (INT8-sim)
      summary.csv               — per-fold metrics (written by aggregate)
      confusion_aggregate.png   — summed confusion matrix over all folds
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.band_selection import select_bands, select_bands_peak
from fbcsp_snn.baseline import extract_logvar, run_baseline_classifiers
from fbcsp_snn.config import Config
from fbcsp_snn.datasets import DATASET_REGISTRY, get_n_classes, load_moabb
from fbcsp_snn.data import load_hdf5
from fbcsp_snn.encoding import encode_tensor
from fbcsp_snn.evaluation import compute_accuracy, compute_confusion_matrix
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier, maybe_compile
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank, window_filter_bank
from fbcsp_snn.quantization import quantize_model, quantization_report
from fbcsp_snn.training import evaluate_model, train_fold
from fbcsp_snn.visualization import (
    plot_band_selection,
    plot_confusion_matrix,
    plot_neuron_traces,
    plot_spike_propagation,
    plot_weight_histograms,
)

logger: logging.Logger = setup_logger(__name__)

# Default class names used when not specified; overridden by dataset registry.
_BNCI2014_CLASSES = ["feet", "left_hand", "right_hand", "tongue"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_names(cfg: Config, n_classes: int) -> list[str]:
    """Return human-readable class labels."""
    if cfg.moabb_dataset == "BNCI2014_001":
        return _BNCI2014_CLASSES[:n_classes]
    return [f"class_{i}" for i in range(n_classes)]


def _load_raw(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw EEG and return (X_train, y_train, X_test, y_test)."""
    if cfg.source == "moabb":
        return load_moabb(cfg.moabb_dataset, cfg.subject_id, cfg.n_classes)
    if cfg.source == "hdf5":
        if cfg.data_path is None:
            logger.error("--data-path required for --source hdf5")
            sys.exit(1)
        X, y = load_hdf5(cfg.data_path)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(sss.split(X, y))
        return X[tr], y[tr], X[te], y[te]
    logger.error("Unknown source: %s", cfg.source)
    sys.exit(1)


def _sfreq(cfg: Config) -> float:
    """Return sampling frequency for the configured dataset."""
    if cfg.source == "moabb" and cfg.moabb_dataset in DATASET_REGISTRY:
        return float(DATASET_REGISTRY[cfg.moabb_dataset]["sfreq"])
    return 250.0


def _spikes_from_concat(
    X_concat: np.ndarray,
    cfg: Config,
) -> torch.Tensor:
    """Encode a concatenated CSP projection array into spikes.

    Parameters
    ----------
    X_concat : np.ndarray
        Shape ``(n_trials, n_features, n_samples)``, float32.
    cfg : Config
        Pipeline config (encoding hyperparameters).

    Returns
    -------
    torch.Tensor
        Binary spikes ``(T, n_trials, n_features)``.
    """
    t = torch.from_numpy(X_concat).to(DEVICE).permute(2, 0, 1)  # (T, B, F)
    return encode_tensor(t, cfg.base_thresh, cfg.adapt_inc, cfg.decay)


def _concat_projections(proj: dict) -> np.ndarray:
    """Concatenate CSP projections from all pairs along the feature axis."""
    return np.concatenate([proj[p] for p in sorted(proj.keys())], axis=1)


def _run_single_fold(
    fold_idx: int,
    X_f_tr: np.ndarray,
    y_f_tr: np.ndarray,
    X_f_val: np.ndarray,
    y_f_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    sfreq: float,
    n_classes: int,
    fold_dir: Path,
) -> dict:
    """Run one CV fold end-to-end and return a metrics dict.

    All preprocessing is fit on the training split only.

    Parameters
    ----------
    fold_idx : int
        0-indexed fold number.
    X_f_tr, y_f_tr : np.ndarray
        Training EEG and labels (1-indexed).
    X_f_val, y_f_val : np.ndarray
        Validation EEG and labels (1-indexed).
    X_test, y_test : np.ndarray
        Held-out test EEG and labels (1-indexed).
    cfg : Config
        Full pipeline config.
    sfreq : float
        Sampling frequency.
    n_classes : int
        Number of MI classes.
    fold_dir : Path
        Directory to save all fold artifacts.

    Returns
    -------
    dict
        Per-fold metrics including val/test FP32 and INT8 accuracies.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    y_f_tr_0  = y_f_tr  - 1
    y_f_val_0 = y_f_val - 1
    y_test_0  = y_test  - 1

    m = cfg.csp_components_per_band // 2   # filters per end

    # ---- Band selection ----
    if cfg.peak_band_selection:
        # Fisher-peak mode: centre one band on each local maximum of the
        # Fisher curve; overlap between bands is explicitly allowed.
        bands, fisher_freqs, fisher_curve = select_bands_peak(
            X_f_tr, y_f_tr, sfreq=sfreq,
            n_bands=cfg.n_adaptive_bands,
            bandwidth=cfg.bandwidth,
            band_range=cfg.band_range,
            min_peak_distance_hz=cfg.peak_min_distance_hz,
            min_fisher_fraction=cfg.min_fisher_fraction,
            top_k_channels=cfg.top_k_channels,
        )
    elif cfg.adaptive_bands:
        # Dense-grid greedy mode (current default).
        bands, fisher_freqs, fisher_curve = select_bands(
            X_f_tr, y_f_tr, sfreq=sfreq,
            n_bands=cfg.n_adaptive_bands,
            bandwidth=cfg.bandwidth,
            step=cfg.band_step,
            band_range=cfg.band_range,
            min_fisher_fraction=cfg.min_fisher_fraction,
            top_k_channels=cfg.top_k_channels,
        )
    else:
        bands = cfg.freq_bands
        fisher_freqs = np.array([0.0])
        fisher_curve = np.array([0.0])

    logger.info("Fold %d  bands: %s", fold_idx, bands)

    plot_band_selection(
        fisher_freqs, fisher_curve, bands,
        save_path=fold_dir / "band_selection.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Band Selection",
    )

    # ---- Filter bank ----
    X_bands_tr  = apply_filter_bank(X_f_tr,  bands, sfreq, order=4)
    X_bands_val = apply_filter_bank(X_f_val, bands, sfreq, order=4)
    X_bands_te  = apply_filter_bank(X_test,  bands, sfreq, order=4)

    # ---- Sliding-window augmentation (CSP fitting only) ----
    # Windows the filtered training bands to increase covariance sample count.
    # Val and test are never touched; CSP spatial filters are applied to
    # full-length trials after fitting.
    if cfg.augment_windows:
        win_samples  = int(cfg.window_duration * sfreq)
        step_samples = int(cfg.window_step * sfreq)
        X_bands_csp, y_csp = window_filter_bank(
            X_bands_tr, y_f_tr, win_samples, step_samples
        )
        n_win = len(y_csp) // len(y_f_tr)
        logger.info(
            "Window augmentation: %d trials × %d windows = %d samples "
            "(window=%.1fs step=%.1fs)",
            len(y_f_tr), n_win, len(y_csp),
            cfg.window_duration, cfg.window_step,
        )
    else:
        X_bands_csp, y_csp = X_bands_tr, y_f_tr

    # ---- Pairwise CSP ----
    csp = PairwiseCSP(m=m, lambda_r=cfg.lambda_r,
                      euclidean_alignment=cfg.euclidean_alignment,
                      riemannian_mean=cfg.riemannian_mean,
                      ledoit_wolf=cfg.csp_ledoit_wolf)
    csp.fit(X_bands_csp, y_csp)

    proj_tr  = csp.transform(X_bands_tr)
    proj_val = csp.transform(X_bands_val)
    proj_te  = csp.transform(X_bands_te)

    X_concat_tr  = _concat_projections(proj_tr)
    X_concat_val = _concat_projections(proj_val)
    X_concat_te  = _concat_projections(proj_te)

    # ---- Z-normalisation ----
    znorm = ZNormaliser()
    X_norm_tr  = znorm.fit_transform(X_concat_tr)
    X_norm_val = znorm.transform(X_concat_val)
    X_norm_te  = znorm.transform(X_concat_te)

    # ---- Classical baseline (log-var + LDA / SVM) --------------------------
    # Runs on the same z-normalised features as the SNN, before spike encoding.
    # Results are stored in pipeline_params.json for direct comparison.
    logger.info("Fold %d  running classical baselines (LDA, SVM) …", fold_idx)
    bl_feat_tr  = extract_logvar(X_norm_tr)
    bl_feat_val = extract_logvar(X_norm_val)
    bl_feat_te  = extract_logvar(X_norm_te)
    baseline_results = run_baseline_classifiers(
        bl_feat_tr,  y_f_tr_0,
        bl_feat_val, y_f_val_0,
        bl_feat_te,  y_test_0,
    )

    # ---- Spike encoding ----
    spikes_tr  = _spikes_from_concat(X_norm_tr,  cfg)
    spikes_val = _spikes_from_concat(X_norm_val, cfg)
    spikes_te  = _spikes_from_concat(X_norm_te,  cfg)

    # ---- MIBIF ----
    mibif: Optional[MIBIFSelector] = None
    if cfg.feature_selection_method == "mibif":
        mibif = MIBIFSelector(
            feature_percentile=cfg.feature_percentile,
            mi_fraction=cfg.mi_fraction,
            random_state=42,
        )
        spikes_tr  = mibif.fit_transform(spikes_tr,  y_f_tr_0)
        spikes_val = mibif.transform(spikes_val)
        spikes_te  = mibif.transform(spikes_te)

    n_input = spikes_tr.shape[2]
    logger.info("Fold %d  n_input_features: %d", fold_idx, n_input)

    # ---- Spike propagation plot (a few training trials) ----
    plot_spike_propagation(
        spikes_tr,
        save_path=fold_dir / "spike_propagation.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Spike Propagation",
        n_trials=4, n_features=min(n_input, 72), t_max=200,
    )

    # ---- Train ----
    model = maybe_compile(SNNClassifier(
        n_input=n_input,
        n_hidden=cfg.hidden_neurons,
        n_classes=n_classes,
        population_per_class=cfg.population_per_class,
        beta=cfg.beta,
        dropout_prob=cfg.dropout_prob,
    ).to(DEVICE))

    result = train_fold(
        spikes_train=spikes_tr,
        y_train=y_f_tr_0,
        spikes_val=spikes_val,
        y_val=y_f_val_0,
        model=model,
        n_classes=n_classes,
        population_per_class=cfg.population_per_class,
        spike_prob=cfg.spiking_prob,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.early_stopping_patience,
        warmup=cfg.early_stopping_warmup,
        tau_vr=10.0,
        batch_size=64,
        device=DEVICE,
        fold_dir=fold_dir,
        log_every=max(1, cfg.epochs // 20),
    )

    # ---- FP32 evaluate ----
    val_acc_fp32,  val_preds_fp32  = evaluate_model(model, spikes_val, y_f_val_0,  DEVICE)
    test_acc_fp32, test_preds_fp32 = evaluate_model(model, spikes_te,  y_test_0,   DEVICE)

    # ---- INT8 simulate + evaluate ----
    model_int8 = quantize_model(model, bits=8)
    val_acc_int8,  val_preds_int8  = evaluate_model(model_int8, spikes_val, y_f_val_0, DEVICE)
    test_acc_int8, test_preds_int8 = evaluate_model(model_int8, spikes_te,  y_test_0,  DEVICE)

    quantization_report(val_acc_fp32,  val_acc_int8,  label="val")
    quantization_report(test_acc_fp32, test_acc_int8, label="test")

    # ---- Neuron traces (one test-set batch) ----
    model.eval()
    with torch.no_grad():
        spk_out_vis, mem_out_vis = model(spikes_te[:, :8, :].to(DEVICE))
    plot_neuron_traces(
        spk_out_vis, mem_out_vis,
        save_path=fold_dir / "neuron_traces.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Output Neuron Traces",
        n_neurons=min(6, n_classes * cfg.population_per_class),
        t_max=200,
    )

    # ---- Weight histograms ----
    plot_weight_histograms(
        model,
        save_path=fold_dir / "weight_histograms.png",
        quantized_model=model_int8,
        title=f"Subject {cfg.subject_id} Fold {fold_idx} — Weight Distributions",
    )

    # ---- Confusion matrices ----
    class_names = _class_names(cfg, n_classes)
    plot_confusion_matrix(
        y_test_0, test_preds_fp32, class_names,
        save_path=fold_dir / "confusion_fp32.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} FP32 — Test",
    )
    plot_confusion_matrix(
        y_test_0, test_preds_int8, class_names,
        save_path=fold_dir / "confusion_int8.png",
        title=f"Subject {cfg.subject_id} Fold {fold_idx} INT8 — Test",
    )

    # ---- Persist preprocessing objects ----
    with open(fold_dir / "csp_filters.pkl", "wb") as f:
        pickle.dump(csp, f, protocol=4)
    with open(fold_dir / "znorm.pkl", "wb") as f:
        pickle.dump(znorm, f, protocol=4)
    if mibif is not None:
        with open(fold_dir / "mibif.pkl", "wb") as f:
            pickle.dump(mibif, f, protocol=4)

    # ---- pipeline_params.json ----
    params = {
        "subject_id":         cfg.subject_id,
        "fold":               fold_idx,
        "dataset":            cfg.moabb_dataset,
        "n_classes":          n_classes,
        "n_input_features":   n_input,
        "bands":              [[float(lo), float(hi)] for lo, hi in bands],
        "adaptive_bands":     cfg.adaptive_bands,
        "euclidean_alignment": cfg.euclidean_alignment,
        "riemannian_mean":    cfg.riemannian_mean,
        "csp_m":              m,
        "lambda_r":           cfg.lambda_r,
        "hidden_neurons":     cfg.hidden_neurons,
        "population_per_class": cfg.population_per_class,
        "beta":               cfg.beta,
        "feature_method":     cfg.feature_selection_method,
        "feature_percentile": cfg.feature_percentile,
        "mi_fraction":        cfg.mi_fraction,
        "n_features_selected": n_input * 2 if mibif is None else len(mibif.selected_indices_),
        "best_val_acc_fp32":  round(result.best_val_acc, 6),
        "best_epoch":         result.best_epoch,
        "stopped_epoch":      result.stopped_epoch,
        "val_acc_fp32":       round(val_acc_fp32, 6),
        "val_acc_int8":       round(val_acc_int8, 6),
        "test_acc_fp32":      round(test_acc_fp32, 6),
        "test_acc_int8":      round(test_acc_int8, 6),
        # Classical baselines (log-var features, same z-norm, no spike encoding)
        "val_acc_lda":        round(baseline_results["val_acc_lda"],  6),
        "test_acc_lda":       round(baseline_results["test_acc_lda"], 6),
        "val_acc_svm":        round(baseline_results["val_acc_svm"],  6),
        "test_acc_svm":       round(baseline_results["test_acc_svm"], 6),
    }
    with open(fold_dir / "pipeline_params.json", "w") as f:
        json.dump(params, f, indent=2)

    logger.info(
        "Fold %d saved to %s  "
        "(FP32 test %.1f%%  INT8 test %.1f%%)",
        fold_idx, fold_dir,
        test_acc_fp32 * 100, test_acc_int8 * 100,
    )
    return params


# ---------------------------------------------------------------------------
# run_train
# ---------------------------------------------------------------------------

def run_train(cfg: Config) -> None:
    """Run the full training pipeline for one subject.

    Executes n_folds CV folds (or a single specified fold).  For each fold:
    band selection → filter bank → CSP → z-norm → spike encoding → MIBIF →
    SNN training → FP32 + INT8 evaluation → artifact save.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration.
    """
    t_start = time.perf_counter()

    # Auto-detect n_classes
    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    logger.info(
        "run_train  subject=%d  dataset=%s  n_classes=%d  n_folds=%d",
        cfg.subject_id, cfg.moabb_dataset, n_classes, cfg.n_folds,
    )

    X_train, y_train, X_test, y_test = _load_raw(cfg)
    sfreq = _sfreq(cfg)

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Determine which folds to run
    # Split session-1 (X_train) only into train/val folds.
    # X_test (session 2) is held out entirely and never touched by the splitter.
    # Default val_fraction=0.2 → StratifiedKFold (80/20, ~58 val trials).
    # Custom val_fraction (e.g. 0.3) → StratifiedShuffleSplit for flexibility.
    if cfg.val_fraction == round(1.0 / cfg.n_folds, 10):
        splitter = StratifiedKFold(
            n_splits=cfg.n_folds, shuffle=True, random_state=42
        )
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=cfg.n_folds, test_size=cfg.val_fraction, random_state=42
        )

    fold_metrics: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        splitter.split(X_train, y_train)
    ):
        if cfg.fold is not None and fold_idx != cfg.fold:
            continue

        logger.info("")
        logger.info("=" * 60)
        logger.info("  Fold %d / %d", fold_idx, cfg.n_folds - 1)
        logger.info("=" * 60)

        fold_dir = subject_dir / f"fold_{fold_idx}"

        metrics = _run_single_fold(
            fold_idx=fold_idx,
            X_f_tr=X_train[tr_idx],
            y_f_tr=y_train[tr_idx],
            X_f_val=X_train[val_idx],
            y_f_val=y_train[val_idx],
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            sfreq=sfreq,
            n_classes=n_classes,
            fold_dir=fold_dir,
        )
        fold_metrics.append(metrics)

    elapsed = time.perf_counter() - t_start
    logger.info("")
    logger.info(
        "run_train complete — %d folds  wall time %.1f s",
        len(fold_metrics), elapsed,
    )

    if fold_metrics:
        mean_test_fp32 = np.mean([m["test_acc_fp32"] for m in fold_metrics])
        mean_test_int8 = np.mean([m["test_acc_int8"] for m in fold_metrics])
        logger.info(
            "Subject %d  mean test FP32 %.1f%%  INT8 %.1f%%",
            cfg.subject_id, mean_test_fp32 * 100, mean_test_int8 * 100,
        )


# ---------------------------------------------------------------------------
# run_infer
# ---------------------------------------------------------------------------

def run_infer(cfg: Config) -> None:
    """Run inference for a single fold using saved artifacts.

    Loads the saved model, CSP, znorm, and MIBIF from ``fold_<K>/``, applies
    the full preprocessing chain to the test set, and logs the test accuracy.

    Parameters
    ----------
    cfg : Config
        Must have ``fold`` set.
    """
    if cfg.fold is None:
        logger.error("--fold is required for infer mode")
        sys.exit(1)

    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    fold_dir = subject_dir / f"fold_{cfg.fold}"

    params_path = fold_dir / "pipeline_params.json"
    if not params_path.exists():
        logger.error("pipeline_params.json not found at %s — run train first", fold_dir)
        sys.exit(1)

    with open(params_path) as f:
        params = json.load(f)

    bands   = [tuple(b) for b in params["bands"]]
    n_input = params["n_input_features"]
    m       = params["csp_m"]
    sfreq   = _sfreq(cfg)

    logger.info(
        "run_infer  subject=%d  fold=%d  bands=%s  n_input=%d",
        cfg.subject_id, cfg.fold, bands, n_input,
    )

    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1

    # Load preprocessing objects
    with open(fold_dir / "csp_filters.pkl", "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(fold_dir / "znorm.pkl", "rb") as f:
        znorm: ZNormaliser = pickle.load(f)

    mibif: Optional[MIBIFSelector] = None
    mibif_path = fold_dir / "mibif.pkl"
    if mibif_path.exists():
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    # Preprocessing chain
    X_bands = apply_filter_bank(X_test, bands, sfreq, order=4)
    proj = csp.transform(X_bands)
    X_concat = _concat_projections(proj)
    X_norm = znorm.transform(X_concat)
    spikes = _spikes_from_concat(X_norm, cfg)
    if mibif is not None:
        spikes = mibif.transform(spikes)

    # Load model
    model = SNNClassifier(
        n_input=n_input,
        n_hidden=params.get("hidden_neurons", cfg.hidden_neurons),
        n_classes=n_classes,
        population_per_class=params.get("population_per_class", cfg.population_per_class),
        beta=params.get("beta", cfg.beta),
        dropout_prob=0.0,   # no dropout at inference
    ).to(DEVICE)
    state = torch.load(fold_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)

    # FP32 inference
    test_acc_fp32, test_preds_fp32 = evaluate_model(model, spikes, y_test_0, DEVICE)

    # INT8-sim inference
    model_int8 = quantize_model(model, bits=8)
    test_acc_int8, test_preds_int8 = evaluate_model(model_int8, spikes, y_test_0, DEVICE)

    quantization_report(test_acc_fp32, test_acc_int8, label="test")

    # Confusion matrices
    class_names = _class_names(cfg, n_classes)
    plot_confusion_matrix(
        y_test_0, test_preds_fp32, class_names,
        save_path=fold_dir / "infer_confusion_fp32.png",
        title=f"Subject {cfg.subject_id} Fold {cfg.fold} FP32 — Inference",
    )
    plot_confusion_matrix(
        y_test_0, test_preds_int8, class_names,
        save_path=fold_dir / "infer_confusion_int8.png",
        title=f"Subject {cfg.subject_id} Fold {cfg.fold} INT8 — Inference",
    )

    logger.info(
        "Fold %d inference — FP32 %.1f%%  INT8 %.1f%%",
        cfg.fold, test_acc_fp32 * 100, test_acc_int8 * 100,
    )


# ---------------------------------------------------------------------------
# run_aggregate
# ---------------------------------------------------------------------------

def run_aggregate(cfg: Config) -> None:
    """Collect per-fold JSON artifacts and produce summary CSV + plots.

    Reads ``fold_K/pipeline_params.json`` for K in ``0 … n_folds-1``,
    writes ``summary.csv``, and saves an aggregated confusion matrix
    (sum over all folds) for FP32 and INT8.

    Parameters
    ----------
    cfg : Config
        Must have ``subject_id``, ``n_folds``, ``results_dir`` set.
    """
    import csv

    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    logger.info("run_aggregate  subject=%d  n_folds=%d", cfg.subject_id, cfg.n_folds)

    rows: list[dict] = []
    missing: list[int] = []

    for fold_idx in range(cfg.n_folds):
        p = subject_dir / f"fold_{fold_idx}" / "pipeline_params.json"
        if not p.exists():
            logger.warning("Missing fold %d  (%s)", fold_idx, p)
            missing.append(fold_idx)
            continue
        with open(p) as f:
            rows.append(json.load(f))

    if not rows:
        logger.error("No fold results found under %s", subject_dir)
        sys.exit(1)

    if missing:
        logger.warning("Aggregating %d / %d folds (missing: %s)",
                       len(rows), cfg.n_folds, missing)

    # ---- Summary CSV ----
    csv_path = subject_dir / "summary.csv"
    fieldnames = [
        "fold", "best_val_acc_fp32", "best_epoch", "stopped_epoch",
        "val_acc_fp32", "val_acc_int8", "test_acc_fp32", "test_acc_int8",
        "val_acc_lda", "test_acc_lda", "val_acc_svm", "test_acc_svm",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Append mean row
    def _col(key: str) -> list[float]:
        return [r[key] for r in rows if key in r]

    mean_row = {
        "fold":              "mean",
        "best_val_acc_fp32": round(float(np.mean(_col("best_val_acc_fp32"))), 6),
        "best_epoch":        "",
        "stopped_epoch":     "",
        "val_acc_fp32":      round(float(np.mean(_col("val_acc_fp32"))),  6),
        "val_acc_int8":      round(float(np.mean(_col("val_acc_int8"))),  6),
        "test_acc_fp32":     round(float(np.mean(_col("test_acc_fp32"))), 6),
        "test_acc_int8":     round(float(np.mean(_col("test_acc_int8"))), 6),
        "val_acc_lda":       round(float(np.mean(_col("val_acc_lda"))),   6) if _col("val_acc_lda")  else "",
        "test_acc_lda":      round(float(np.mean(_col("test_acc_lda"))),  6) if _col("test_acc_lda") else "",
        "val_acc_svm":       round(float(np.mean(_col("val_acc_svm"))),   6) if _col("val_acc_svm")  else "",
        "test_acc_svm":      round(float(np.mean(_col("test_acc_svm"))),  6) if _col("test_acc_svm") else "",
    }
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(mean_row)

    logger.info("Summary CSV written: %s", csv_path)

    # Log table to stdout
    logger.info("")
    logger.info(
        "  %-6s  %-12s  %-12s  %-10s  %-10s",
        "Fold", "TestFP32", "TestINT8", "LDA", "SVM",
    )
    logger.info("  " + "-" * 58)
    for r in rows:
        logger.info(
            "  %-6s  %-12.1f  %-12.1f  %-10.1f  %-10.1f",
            r["fold"],
            r["test_acc_fp32"]  * 100,
            r["test_acc_int8"]  * 100,
            r.get("test_acc_lda", 0) * 100,
            r.get("test_acc_svm", 0) * 100,
        )
    logger.info("  " + "-" * 58)
    logger.info(
        "  %-6s  %-12.1f  %-12.1f  %-10.1f  %-10.1f",
        "MEAN",
        mean_row["test_acc_fp32"]  * 100,   # type: ignore[operator]
        mean_row["test_acc_int8"]  * 100,   # type: ignore[operator]
        (mean_row["test_acc_lda"]  * 100) if mean_row.get("test_acc_lda") else 0.0,
        (mean_row["test_acc_svm"]  * 100) if mean_row.get("test_acc_svm") else 0.0,
    )

    # ---- Aggregated confusion matrix ----
    # Re-load test predictions by re-running inference on each fold's saved model.
    # This is lightweight since we just decode; no retraining.
    class_names = _class_names(cfg, n_classes)
    cm_fp32_sum = np.zeros((n_classes, n_classes), dtype=float)
    cm_int8_sum = np.zeros((n_classes, n_classes), dtype=float)

    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1
    sfreq = _sfreq(cfg)

    for r in rows:
        fold_idx = r["fold"]
        fold_dir = subject_dir / f"fold_{fold_idx}"
        params   = r

        bands   = [tuple(b) for b in params["bands"]]
        n_input = params["n_input_features"]

        # Load preprocessing
        try:
            with open(fold_dir / "csp_filters.pkl", "rb") as f:
                csp: PairwiseCSP = pickle.load(f)
            with open(fold_dir / "znorm.pkl", "rb") as f:
                znorm: ZNormaliser = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Preprocessing pickle missing for fold %s — skip", fold_idx)
            continue

        mibif_path = fold_dir / "mibif.pkl"
        mibif: Optional[MIBIFSelector] = None
        if mibif_path.exists():
            with open(mibif_path, "rb") as f:
                mibif = pickle.load(f)

        X_bands = apply_filter_bank(X_test, bands, sfreq, order=4)
        proj    = csp.transform(X_bands)
        X_concat = _concat_projections(proj)
        X_norm  = znorm.transform(X_concat)
        spikes  = _spikes_from_concat(X_norm, cfg)
        if mibif is not None:
            spikes = mibif.transform(spikes)

        model_path = fold_dir / "best_model.pt"
        if not model_path.exists():
            logger.warning("best_model.pt missing for fold %s — skip", fold_idx)
            continue

        model = SNNClassifier(
            n_input=n_input,
            n_hidden=params.get("hidden_neurons", cfg.hidden_neurons),
            n_classes=n_classes,
            population_per_class=params.get("population_per_class", cfg.population_per_class),
            beta=params.get("beta", cfg.beta),
            dropout_prob=0.0,
        ).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)

        _, preds_fp32 = evaluate_model(model, spikes, y_test_0, DEVICE)
        cm_fp32_sum += compute_confusion_matrix(
            y_test_0, preds_fp32, n_classes=n_classes, normalize=False
        )

        model_int8 = quantize_model(model, bits=8)
        _, preds_int8 = evaluate_model(model_int8, spikes, y_test_0, DEVICE)
        cm_int8_sum += compute_confusion_matrix(
            y_test_0, preds_int8, n_classes=n_classes, normalize=False
        )

    # Row-normalise the summed matrices
    def _row_norm(cm: np.ndarray) -> np.ndarray:
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sums, where=row_sums > 0)

    from fbcsp_snn.visualization import plot_confusion_matrix as _plot_cm

    _plot_cm(
        np.repeat(np.arange(n_classes), int(cm_fp32_sum.sum() / n_classes + 1))[:int(cm_fp32_sum.sum())],
        np.zeros(int(cm_fp32_sum.sum()), dtype=int),   # dummy — not used, matrix supplied directly
        class_names,
        save_path=subject_dir / "confusion_aggregate_fp32.png",
        title=f"Subject {cfg.subject_id} — Aggregated FP32 ({len(rows)} folds)",
    ) if False else None   # handled below via direct imshow

    _save_cm_from_array(
        _row_norm(cm_fp32_sum), class_names,
        save_path=subject_dir / "confusion_aggregate_fp32.png",
        title=f"Subject {cfg.subject_id} — Aggregated FP32 ({len(rows)} folds)",
    )
    _save_cm_from_array(
        _row_norm(cm_int8_sum), class_names,
        save_path=subject_dir / "confusion_aggregate_int8.png",
        title=f"Subject {cfg.subject_id} — Aggregated INT8 ({len(rows)} folds)",
    )

    logger.info("Aggregation complete for Subject %d.", cfg.subject_id)


def _save_cm_from_array(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str,
) -> None:
    """Save a pre-computed normalised confusion matrix as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3.5, n)))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
        vmin=0.0, vmax=1.0,
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Aggregated confusion matrix saved: %s", save_path)
