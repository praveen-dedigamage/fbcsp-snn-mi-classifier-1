"""Pipeline reliability / hardware-realism testing (B15).

Runs Monte Carlo noise-robustness sweeps on an already-trained fold's saved
checkpoint -- no retraining. Tests three analog non-idealities (CSP weight
noise, SNN weight noise, LIF beta/leak-rate variation) at multiple
severities, each repeated N times with fresh random draws, reporting
mean +/- std accuracy and mean spike-events-per-trial per condition.

This is distinct from -- and answers a different question than -- the
deterministic bit-quantization sweep already baked into training (see
``pipeline.py``'s PTQ and joint-PTQ blocks). Bit quantization asks "how many
bits does this need"; this module asks "does this survive the kind of noise
an actual analog circuit has" (thermal noise, device mismatch, comparator
offset, RC time-constant variation) -- the deterministic bit-rounding used
elsewhere models none of that.

Usage::

    python main.py reliability --subject-id 1 --fold 0 \\
        --reliability-severities "[0.0,0.05,0.1,0.2,0.3]" \\
        --reliability-n-repeats 20
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.config import Config
from fbcsp_snn.datasets import get_n_classes
from fbcsp_snn.mibif import MIBIFSelector
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import PairwiseCSP, ZNormaliser, apply_filter_bank
from fbcsp_snn.quantization import (
    inject_beta_noise,
    inject_csp_filter_noise,
    inject_model_weight_noise,
)
from fbcsp_snn.training import evaluate_model_with_events

logger: logging.Logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo aggregation (B15 Tier C)
# ---------------------------------------------------------------------------

def monte_carlo_eval(
    eval_fn: Callable[[int], Tuple[float, float]],
    n_repeats: int,
) -> Dict[str, float]:
    """Repeat a noisy evaluation with distinct seeds; aggregate mean +/- std.

    Since analog noise is stochastic, a single deterministic run (as the
    existing bit-quantization sweep does) doesn't characterise robustness --
    every noisy condition needs multiple independent draws.

    Parameters
    ----------
    eval_fn : Callable[[int], Tuple[float, float]]
        Takes a seed, returns ``(accuracy, mean_events_per_trial)`` for one
        noisy draw.
    n_repeats : int
        Number of Monte Carlo repeats with distinct seeds.

    Returns
    -------
    Dict[str, float]
        ``acc_mean``, ``acc_std``, ``events_mean``, ``events_std``.
    """
    accs: List[float] = []
    events: List[float] = []
    for seed in range(n_repeats):
        acc, ev = eval_fn(seed)
        accs.append(acc)
        events.append(ev)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "events_mean": float(np.mean(events)),
        "events_std": float(np.std(events)),
    }


# ---------------------------------------------------------------------------
# Full reliability sweep for one saved fold
# ---------------------------------------------------------------------------

def run_reliability(cfg: Config) -> None:
    """Run the full noise-robustness sweep for one saved fold (B15).

    Loads the fold's saved CSP/znorm/MIBIF/model artifacts (same loading
    logic as ``run_infer``) and, for each of three noise sources at each
    severity in ``cfg.reliability_severities``, runs
    ``cfg.reliability_n_repeats`` Monte Carlo evaluations and aggregates
    mean +/- std accuracy and mean events/trial. Results are saved to
    ``fold_dir/reliability_results.json``. No retraining occurs.

    Parameters
    ----------
    cfg : Config
        Must have ``fold`` set; ``reliability_severities`` and
        ``reliability_n_repeats`` control the sweep grid.
    """
    if cfg.fold is None:
        logger.error("--fold is required for reliability mode")
        raise SystemExit(1)

    # Local import: reliability.py reuses pipeline.py's internal data-loading
    # helpers rather than duplicating them. One-directional (pipeline.py does
    # not import from here), so no circular-import risk.
    from fbcsp_snn.pipeline import (
        _concat_projections,
        _load_raw,
        _sfreq,
        _spikes_from_concat,
    )

    if cfg.n_classes is None and cfg.source == "moabb":
        cfg.n_classes = get_n_classes(cfg.moabb_dataset)
    n_classes: int = cfg.n_classes  # type: ignore[assignment]

    subject_dir = Path(cfg.results_dir) / f"Subject_{cfg.subject_id}"
    fold_dir = subject_dir / f"fold_{cfg.fold}"

    params_path = fold_dir / "pipeline_params.json"
    if not params_path.exists():
        logger.error("pipeline_params.json not found at %s -- run train first", fold_dir)
        raise SystemExit(1)
    with open(params_path) as f:
        params = json.load(f)

    bands = [tuple(b) for b in params["bands"]]
    n_input = params["n_input_features"]
    sfreq = _sfreq(cfg)

    logger.info(
        "run_reliability  subject=%d  fold=%d  severities=%s  n_repeats=%d",
        cfg.subject_id, cfg.fold, cfg.reliability_severities, cfg.reliability_n_repeats,
    )

    _, _, X_test, y_test = _load_raw(cfg)
    y_test_0 = y_test - 1

    with open(fold_dir / "csp_filters.pkl", "rb") as f:
        csp: PairwiseCSP = pickle.load(f)
    with open(fold_dir / "znorm.pkl", "rb") as f:
        znorm: ZNormaliser = pickle.load(f)

    mibif: Optional[MIBIFSelector] = None
    mibif_path = fold_dir / "mibif.pkl"
    if mibif_path.exists():
        with open(mibif_path, "rb") as f:
            mibif = pickle.load(f)

    filter_type = params.get("filter_type", "butterworth")
    cfg.encoder_type = params.get("encoder_type", cfg.encoder_type)

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

    saved_csp_filters = csp.filters_

    def _encode_test(csp_filters_to_use: dict) -> torch.Tensor:
        """Re-run the preprocessing chain on test data with the given CSP filters."""
        csp.filters_ = csp_filters_to_use
        X_bands = apply_filter_bank(X_test, bands, sfreq, order=4, filter_type=filter_type)
        proj = csp.transform(X_bands)
        X_concat = _concat_projections(proj)
        X_norm = znorm.transform(X_concat)
        spikes = _spikes_from_concat(X_norm, cfg)
        if mibif is not None:
            spikes = mibif.transform(spikes)
        return spikes

    # ---- Clean (noise-free) baseline ----
    spikes_clean = _encode_test(saved_csp_filters)
    clean_acc, _, clean_events = evaluate_model_with_events(
        model, spikes_clean, y_test_0, DEVICE
    )
    logger.info("Clean baseline -- acc: %.4f  events/trial: %.1f", clean_acc, clean_events)

    severities = cfg.reliability_severities
    n_repeats = cfg.reliability_n_repeats
    results: Dict[str, object] = {
        "clean": {"acc": clean_acc, "events_per_trial": clean_events},
        "n_repeats": n_repeats,
        "severities": severities,
    }

    def _sweep(name: str, eval_at_sigma: Callable[[float, int], Tuple[float, float]]) -> None:
        sweep_results: Dict[str, dict] = {}
        for sigma in severities:
            if sigma == 0.0:
                sweep_results[str(sigma)] = {
                    "acc_mean": clean_acc, "acc_std": 0.0,
                    "events_mean": clean_events, "events_std": 0.0,
                }
                continue
            stats = monte_carlo_eval(lambda seed, s=sigma: eval_at_sigma(s, seed), n_repeats)
            sweep_results[str(sigma)] = stats
            logger.info(
                "%s  sigma=%.3f  acc %.4f+-%.4f  events %.1f+-%.1f",
                name, sigma, stats["acc_mean"], stats["acc_std"],
                stats["events_mean"], stats["events_std"],
            )
        results[name] = sweep_results

    # ---- CSP weight noise (conductance variation in an analog crossbar) ----
    def _eval_csp_noise(sigma: float, seed: int) -> Tuple[float, float]:
        noisy_filters = inject_csp_filter_noise(saved_csp_filters, sigma, seed=seed)
        spikes_noisy = _encode_test(noisy_filters)
        acc, _, ev = evaluate_model_with_events(model, spikes_noisy, y_test_0, DEVICE)
        return acc, ev
    _sweep("csp_weight_noise", _eval_csp_noise)
    csp.filters_ = saved_csp_filters  # restore before returning

    # ---- SNN weight noise (conductance variation in the classifier stage) ----
    def _eval_snn_noise(sigma: float, seed: int) -> Tuple[float, float]:
        model_noisy = inject_model_weight_noise(model, sigma, seed=seed)
        acc, _, ev = evaluate_model_with_events(model_noisy, spikes_clean, y_test_0, DEVICE)
        return acc, ev
    _sweep("snn_weight_noise", _eval_snn_noise)

    # ---- Beta (leak-rate) variation (RC time-constant mismatch) ----
    def _eval_beta_noise(sigma: float, seed: int) -> Tuple[float, float]:
        model_noisy = inject_beta_noise(model, sigma, seed=seed)
        acc, _, ev = evaluate_model_with_events(model_noisy, spikes_clean, y_test_0, DEVICE)
        return acc, ev
    _sweep("beta_noise", _eval_beta_noise)

    out_path = fold_dir / "reliability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Reliability sweep saved to %s", out_path)
