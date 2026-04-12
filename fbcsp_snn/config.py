"""Config dataclass and argparse CLI for the FBCSP-SNN pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    """All pipeline hyperparameters and runtime options.

    Parameters
    ----------
    mode : str
        Pipeline mode: ``"train"``, ``"infer"``, or ``"aggregate"``.
    source : str
        Data source: ``"moabb"`` or ``"hdf5"``.
    moabb_dataset : str
        MOABB dataset name (e.g. ``"BNCI2014_001"``).
    subject_id : int
        Subject index (1-indexed).
    data_path : Optional[str]
        Path to HDF5 .mat file (only used when ``source="hdf5"``).
    fold : Optional[int]
        Single fold to run (``None`` → run all folds).
    n_folds : int
        Number of CV folds.
    adaptive_bands : bool
        Use adaptive frequency band selection.
    n_adaptive_bands : int
        Number of bands to select adaptively.
    freq_bands : List[Tuple[float, float]]
        Static frequency bands (used when ``adaptive_bands=False``).
    bandwidth : float
        Candidate band width in Hz.
    band_step : float
        Candidate band step in Hz.
    band_range : Tuple[float, float]
        Frequency range for candidate bands.
    csp_components_per_band : int
        Total CSP filters per band (half from each end).
    lambda_r : float
        CSP covariance regularisation.
    base_thresh : float
        Spike encoding base threshold.
    adapt_inc : float
        Spike encoding adaptive increment.
    decay : float
        Spike encoding threshold decay.
    hidden_neurons : int
        Hidden layer size of the SNN.
    population_per_class : int
        Output population neurons per class.
    beta : float
        LIF membrane potential decay factor.
    dropout_prob : float
        Dropout probability.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    epochs : int
        Maximum training epochs.
    early_stopping_patience : int
        Early stopping patience (epochs).
    early_stopping_warmup : int
        Minimum epochs before early stopping kicks in.
    spiking_prob : float
        Target class spike probability for Van Rossum loss.
    feature_selection_method : str
        Feature selection method (``"mibif"`` or ``"none"``).
    feature_percentile : float
        Percentile of features to keep.
    results_dir : str
        Root directory for output artifacts.
    n_classes : Optional[int]
        Number of classes (auto-detected from dataset if ``None``).
    """

    # Runtime
    mode: str = "train"
    source: str = "moabb"
    moabb_dataset: str = "BNCI2014_001"
    subject_id: int = 1
    data_path: Optional[str] = None
    fold: Optional[int] = None
    n_folds: int = 10
    val_fraction: float = 0.2

    # Band selection
    adaptive_bands: bool = True
    n_adaptive_bands: int = 12
    freq_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(4, 8), (8, 14), (14, 30)]
    )
    bandwidth: float = 4.0
    band_step: float = 2.0
    band_range: Tuple[float, float] = (4.0, 40.0)
    min_fisher_fraction: float = 0.15
    peak_band_selection: bool = False
    peak_min_distance_hz: float = 2.0
    top_k_channels: Optional[int] = None
    channel_specific_bands: bool = False

    # CSP
    csp_components_per_band: int = 8
    lambda_r: float = 0.0001
    euclidean_alignment: bool = True
    riemannian_mean: bool = True
    csp_ledoit_wolf: bool = False

    # Data augmentation (CSP fitting only)
    augment_windows: bool = False
    window_duration: float = 2.0   # seconds
    window_step: float = 0.5       # seconds → 75 % overlap at 250 Hz
    freq_shift_augment: bool = False
    freq_shift_hz: float = 2.0     # ± Hz shift for frequency-shift augmentation

    # Encoding
    base_thresh: float = 0.001
    adapt_inc: float = 0.6
    decay: float = 0.95

    # Model
    hidden_neurons: int = 64
    population_per_class: int = 20
    beta: float = 0.95
    dropout_prob: float = 0.5

    # Training
    lr: float = 1e-3
    weight_decay: float = 0.1
    epochs: int = 1000
    early_stopping_patience: int = 100
    early_stopping_warmup: int = 100
    spiking_prob: float = 0.7

    # Feature selection
    feature_selection_method: str = "mibif"
    feature_percentile: float = 50.0
    mi_fraction: Optional[float] = None   # adaptive mode: keep MI >= mi_fraction*max_MI

    # Epoch window (passed to MOABB MotorImagery paradigm)
    tmin: float = 0.0   # seconds after cue onset
    tmax: float = 3.0   # seconds after cue onset

    # I/O
    results_dir: str = "Results"
    n_classes: Optional[int] = None


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------

def _parse_freq_bands(value: str) -> List[Tuple[float, float]]:
    """Parse a string like ``"[(4,10),(10,14),(14,30)]"`` into a list of tuples."""
    import ast
    parsed = ast.literal_eval(value)
    return [(float(lo), float(hi)) for lo, hi in parsed]


def _parse_band_range(value: str) -> Tuple[float, float]:
    """Parse a band-range string into a ``(lo, hi)`` tuple.

    Accepts three formats:
    - ``"4.0:30.0"``    — colon-separated (shell-safe, SLURM --export safe)
    - ``"(4.0,30.0)"``  — tuple literal
    - ``"[4.0,30.0]"``  — list literal
    """
    if ":" in value:
        parts = value.split(":")
        return (float(parts[0]), float(parts[1]))
    import ast
    lo, hi = ast.literal_eval(value)
    return (float(lo), float(hi))


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FBCSP-SNN Motor Imagery EEG Classifier",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- shared arguments ----
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--source", choices=["moabb", "hdf5"], default="moabb")
    shared.add_argument("--moabb-dataset", default="BNCI2014_001")
    shared.add_argument("--subject-id", type=int, default=1)
    shared.add_argument("--data-path", default=None)
    shared.add_argument("--results-dir", default="Results")
    shared.add_argument("--n-classes", type=int, default=None)

    # ---- train ----
    train_p = sub.add_parser("train", parents=[shared], help="Train the pipeline.")
    train_p.add_argument("--n-folds", type=int, default=10)
    train_p.add_argument("--val-fraction", type=float, default=0.2,
                         help="Validation fraction per fold (default 0.2 = 80/20 split).")
    train_p.add_argument("--fold", type=int, default=None,
                         help="Run only this fold (0-indexed).")
    train_p.add_argument("--tmin", type=float, default=0.0,
                         help="Epoch start in seconds after cue onset (default 0.0).")
    train_p.add_argument("--tmax", type=float, default=3.0,
                         help="Epoch end in seconds after cue onset (default 3.0).")
    train_p.add_argument("--adaptive-bands", action="store_true", default=True)
    train_p.add_argument("--no-adaptive-bands", dest="adaptive_bands",
                         action="store_false")
    train_p.add_argument("--n-adaptive-bands", type=int, default=6)
    train_p.add_argument("--freq-bands", type=_parse_freq_bands,
                         default=[(4, 8), (8, 14), (14, 30)])
    train_p.add_argument("--band-range", type=_parse_band_range,
                         default=(4.0, 40.0),
                         help="Candidate band search range, e.g. '(4.0,30.0)'")
    train_p.add_argument("--min-fisher-fraction", type=float, default=0.05,
                         help="Min Fisher score as fraction of top band score (default 0.05).")
    train_p.add_argument("--bandwidth", type=float, default=4.0)
    train_p.add_argument("--band-step", type=float, default=2.0)
    train_p.add_argument("--peak-band-selection", dest="peak_band_selection",
                         action="store_true", default=False,
                         help="Select bands by centering on Fisher-ratio peaks "
                              "(overlap allowed) instead of the dense-grid greedy method.")
    train_p.add_argument("--peak-min-distance-hz", type=float, default=2.0,
                         help="Minimum separation between peak centres in Hz (default 2.0).")
    train_p.add_argument("--top-k-channels", type=int, default=None,
                         help="Use only the top-K most discriminative channels (by peak Fisher "
                              "ratio) when computing the Fisher curve for band selection. "
                              "None (default) uses all channels.")
    train_p.add_argument("--channel-specific-bands", dest="channel_specific_bands",
                         action="store_true", default=False,
                         help="Approach B: filter each EEG channel at its own Fisher-peak "
                              "centre frequency instead of a shared global band. Produces "
                              "n_adaptive_bands channel-personalised filter-bank slots.")
    train_p.add_argument("--csp-components-per-band", type=int, default=4)
    train_p.add_argument("--lambda-r", type=float, default=0.0001)
    train_p.add_argument("--euclidean-alignment", action="store_true", default=True)
    train_p.add_argument("--no-euclidean-alignment", dest="euclidean_alignment",
                         action="store_false")
    train_p.add_argument("--riemannian-mean", action="store_true", default=True)
    train_p.add_argument("--no-riemannian-mean", dest="riemannian_mean",
                         action="store_false")
    train_p.add_argument("--csp-ledoit-wolf", dest="csp_ledoit_wolf",
                         action="store_true", default=False,
                         help="Use Ledoit-Wolf shrinkage for CSP covariance "
                              "estimation instead of fixed Tikhonov regularisation.")
    train_p.add_argument("--augment-windows", dest="augment_windows",
                         action="store_true", default=False,
                         help="Augment CSP covariance fitting with overlapping "
                              "sliding windows (val/test unaffected).")
    train_p.add_argument("--window-duration", type=float, default=2.0,
                         help="Sliding window length in seconds (default 2.0).")
    train_p.add_argument("--window-step", type=float, default=0.5,
                         help="Sliding window step in seconds (default 0.5 → "
                              "75%% overlap at 250 Hz).")
    train_p.add_argument("--freq-shift-augment", dest="freq_shift_augment",
                         action="store_true", default=False,
                         help="Augment CSP fitting data with ±freq_shift_hz copies "
                              "of each band to simulate inter-session spectral drift.")
    train_p.add_argument("--freq-shift-hz", type=float, default=2.0,
                         help="Frequency shift magnitude in Hz (default 2.0).")
    train_p.add_argument("--base-thresh", type=float, default=0.001)
    train_p.add_argument("--adapt-inc", type=float, default=0.6)
    train_p.add_argument("--decay", type=float, default=0.95)
    train_p.add_argument("--hidden-neurons", type=int, default=64)
    train_p.add_argument("--population-per-class", type=int, default=20)
    train_p.add_argument("--beta", type=float, default=0.95)
    train_p.add_argument("--dropout-prob", type=float, default=0.5)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=0.1)
    train_p.add_argument("--epochs", type=int, default=1000)
    train_p.add_argument("--early-stopping-patience", type=int, default=100)
    train_p.add_argument("--early-stopping-warmup", type=int, default=100)
    train_p.add_argument("--spiking-prob", type=float, default=0.7)
    train_p.add_argument("--feature-selection-method",
                         choices=["mibif", "none"], default="mibif")
    train_p.add_argument("--feature-percentile", type=float, default=50.0)
    train_p.add_argument("--mi-fraction", type=float, default=None,
                         help="Adaptive MIBIF threshold: keep features with "
                              "MI >= mi_fraction * max_MI. When set, overrides "
                              "--feature-percentile. Try 0.05-0.3.")

    # ---- infer ----
    infer_p = sub.add_parser("infer", parents=[shared], help="Run inference.")
    infer_p.add_argument("--fold", type=int, required=True)
    infer_p.add_argument("--n-folds", type=int, default=10)

    # ---- aggregate ----
    agg_p = sub.add_parser("aggregate", parents=[shared],
                            help="Aggregate fold results.")
    agg_p.add_argument("--n-folds", type=int, default=10)

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Convert parsed ``argparse.Namespace`` to a :class:`Config` instance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    Config
        Populated config object.
    """
    kwargs: dict = {}
    mapping = {
        "mode": "mode",
        "source": "source",
        "moabb_dataset": "moabb_dataset",
        "subject_id": "subject_id",
        "data_path": "data_path",
        "results_dir": "results_dir",
        "n_classes": "n_classes",
    }
    for attr, field_name in mapping.items():
        if hasattr(args, attr):
            kwargs[field_name] = getattr(args, attr)

    # mode-specific fields (all optional — Config has defaults)
    optional_fields = [
        "n_folds", "fold", "val_fraction", "tmin", "tmax",
        "adaptive_bands", "n_adaptive_bands", "freq_bands",
        "band_range", "bandwidth", "band_step", "min_fisher_fraction",
        "peak_band_selection", "peak_min_distance_hz", "top_k_channels",
        "channel_specific_bands",
        "csp_components_per_band", "lambda_r", "euclidean_alignment", "riemannian_mean", "csp_ledoit_wolf",
        "augment_windows", "window_duration", "window_step",
        "freq_shift_augment", "freq_shift_hz",
        "base_thresh", "adapt_inc", "decay",
        "hidden_neurons", "population_per_class", "beta", "dropout_prob",
        "lr", "weight_decay", "epochs", "early_stopping_patience",
        "early_stopping_warmup", "spiking_prob",
        "feature_selection_method", "feature_percentile", "mi_fraction",
    ]
    for f in optional_fields:
        if hasattr(args, f):
            kwargs[f] = getattr(args, f)

    return Config(**kwargs)
