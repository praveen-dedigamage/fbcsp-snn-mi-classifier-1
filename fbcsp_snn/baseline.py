"""Classical FBCSP baseline classifiers.

Extracts log-variance features from z-normalised CSP projections and
evaluates LDA and SVM classifiers on the same train/val/test splits as
the SNN.  These results serve as an upper-bound reference: if LDA/SVM
can't beat the SNN's accuracy, the bottleneck is in feature extraction,
not the spiking model.

Typical usage (called from pipeline._run_single_fold after z-norm):

    from fbcsp_snn.baseline import extract_logvar, run_baseline_classifiers
    feat_tr  = extract_logvar(X_norm_tr)
    feat_val = extract_logvar(X_norm_val)
    feat_te  = extract_logvar(X_norm_te)
    bl = run_baseline_classifiers(feat_tr, y_tr, feat_val, y_val, feat_te, y_te)
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fbcsp_snn import setup_logger

logger = setup_logger(__name__)


def extract_logvar(X_norm: np.ndarray) -> np.ndarray:
    """Compute log-variance features from z-normalised CSP projections.

    Parameters
    ----------
    X_norm : np.ndarray
        Z-normalised CSP projections, shape ``(n_trials, n_features, n_samples)``.

    Returns
    -------
    np.ndarray
        Log-variance features, shape ``(n_trials, n_features)``.
    """
    var = np.var(X_norm, axis=-1)          # (n_trials, n_features)
    var = np.clip(var, 1e-10, None)        # guard against log(0)
    return np.log(var).astype(np.float32)


def run_baseline_classifiers(
    feat_tr:  np.ndarray,
    y_tr:     np.ndarray,
    feat_val: np.ndarray,
    y_val:    np.ndarray,
    feat_te:  np.ndarray,
    y_te:     np.ndarray,
) -> dict[str, float]:
    """Train LDA and SVM on log-variance features and evaluate on all splits.

    Both classifiers are fit on ``feat_tr / y_tr`` only.  No hyperparameter
    search is performed — default settings match the standard MI-BCI literature.

    Parameters
    ----------
    feat_tr : np.ndarray
        Training log-variance features, shape ``(n_trials, n_features)``.
    y_tr : np.ndarray
        Training class labels, 0-indexed, shape ``(n_trials,)``.
    feat_val : np.ndarray
        Validation features, shape ``(n_trials, n_features)``.
    y_val : np.ndarray
        Validation labels, 0-indexed.
    feat_te : np.ndarray
        Test features, shape ``(n_trials, n_features)``.
    y_te : np.ndarray
        Test labels, 0-indexed.

    Returns
    -------
    dict[str, float]
        ``val_acc_lda``, ``test_acc_lda``, ``val_acc_svm``, ``test_acc_svm``
        — all accuracies in ``[0, 1]``.
    """
    results: dict[str, float] = {}

    # ---- LDA ----------------------------------------------------------------
    # Standard choice for FBCSP in MI-BCI (same as MNE-Python default pipeline).
    # solver='svd' avoids computing the full covariance matrix — more stable
    # when n_features > n_trials.
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(feat_tr, y_tr)
    results["val_acc_lda"]  = float(lda.score(feat_val, y_val))
    results["test_acc_lda"] = float(lda.score(feat_te,  y_te))
    logger.info(
        "Baseline LDA   val=%.1f%%  test=%.1f%%",
        results["val_acc_lda"] * 100, results["test_acc_lda"] * 100,
    )

    # ---- SVM (RBF) ----------------------------------------------------------
    # RBF-SVM with StandardScaler is consistently competitive on CSP features.
    # C=1, gamma='scale' are sensible defaults without cross-validated tuning.
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)),
    ])
    svm.fit(feat_tr, y_tr)
    results["val_acc_svm"]  = float(svm.score(feat_val, y_val))
    results["test_acc_svm"] = float(svm.score(feat_te,  y_te))
    logger.info(
        "Baseline SVM   val=%.1f%%  test=%.1f%%",
        results["val_acc_svm"] * 100, results["test_acc_svm"] * 100,
    )

    return results
