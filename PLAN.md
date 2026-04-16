# Publication Plan — FBCSP-SNN Neuromorphic EEG Classifier

Generated with Opus 4.6. Work through items in order; Priority 1 items are non-negotiable
for a publishable paper.

---

## Priority 1 — Required for a publishable paper

Without these, the paper either can't be written or won't survive review.

- [x] **1. Bessel-filter experiment.** ✓ CLOSED 2026-04-16
  Result: 63.4% ±14.0 — −2.8 pp vs causal-Butterworth (66.2%), −3.8 pp vs zero-phase (67.2%).
  Flat group-delay hypothesis failed; S7 still regressed (69.3 vs 70.9). Only S2 (50.8%, best
  ever) and S4 (64.0%) benefit. Causal Butterworth remains the neuromorphic-compatible choice.

- [x] **2. Adaptive ADM encoder.** ✓ CLOSED 2026-04-16
  Result: 67.4% ±15.2 — +0.2pp vs static6-overlap (67.2%), new best.
  S2 +6.7pp, S9 +3.5pp, S1 +1.7pp drive the gain; S6 −4.2pp main loser.
  ADM adds direct silicon precedent (Lichtsteiner & Liu address-event camera).

- [x] **3. Persistent-state flag.** ✓ RESOLVED 2026-04-16 — no code change needed.
  True persistent state across trial boundaries is invalid for epoched MI data: the paradigm
  includes non-MI cues and rest periods between imagery windows that the pipeline never sees.
  Connecting end of MI trial N directly to start of trial N+1 creates fake continuity.
  Resolution (option 2): each primitive is individually streaming-compatible (causal IIR zi,
  ADM v_ref, LIF V). Scoped honestly in the methods section — benchmarked on epoched data
  per BNCI2014_001 protocol, full-stream deployment left for hardware validation (item #8).

- [x] **4. ADM A/B sweep across all 9 subjects, 5 folds.** ✓ CLOSED 2026-04-16
  Result: 67.4% ±15.2 — +0.2pp vs delta encoder (67.2%). Parity check passed.
  Completed as part of item #2 (Results_adm_static6 run).

- [ ] **5. CSP weight quantization sweep.**
  Deliverable: accuracy-vs-precision cliff plot. Target: ≤1 pp drop at 6-bit.

  Pre-analysis complete (2026-04-16) — `analyze_csp_weights.py` on Results_adm_static6:
  - 4-bit: NRMSE 20–29%, SQNR 10–14 dB — analytically too coarse (step ≈ 1×std)
  - 6-bit: NRMSE 4.7–6.4%, SQNR 24–27 dB — borderline, S6 worst (outlier-driven)
  - 8-bit: NRMSE 1.1–1.6%, SQNR 36–39 dB — safe for all subjects
  - S6 has the heaviest outliers (abs_max=7.4 vs p99=2.7) — drives worst-case at all levels
  - Keep 4-bit as cliff marker even though it will fail

  Subtasks:

  - [ ] **5a. Implement --csp-bits flag (PTQ inference path).**
    Add symmetric per-tensor quantization to CSP weights after loading from pickle,
    before projection. Reuse existing trained models from Results_adm_static6 — no
    retraining. ~30 lines in quantization.py + pipeline.py.

  - [ ] **5b. Run PTQ inference sweep (4, 6, 8-bit) on Puhti.**
    Re-run inference only on all 9 subjects × 5 folds at each bit-width using
    existing trained models. Command: `python main.py infer --csp-bits N ...`
    Deliverable: accuracy table per bit-width.

  - [ ] **5c. Evaluate PTQ results — retrain decision.**
    If 6-bit PTQ drops ≤1 pp → done, paper claim holds.
    If 6-bit PTQ drops >1 pp → proceed to 5d (QAT).

  - [ ] **5d. [Conditional] QAT — retrain SNN with quantized CSP weights.**
    Only if 5c fails. Add --csp-bits to training path. 3 × 45 = 135 Puhti jobs.
    Answers: "SNN trained and deployed under hardware-precision constraints."

- [ ] **6. Butterworth coefficient sensitivity (Monte Carlo).**
  Perturb each SOS coefficient with Gaussian noise at σ = 1%, 2%, 5%, 100 draws per level.
  Report accuracy distribution.
  Effort: ~80 lines for the perturbation harness, single Puhti submit per σ level. ~3 days.

- [ ] **7. End-to-end hardware-constrained accuracy.**
  Combine worst-acceptable filter mismatch (from #6) with worst-acceptable CSP precision
  (from #5) and the existing INT8 SNN. Run full pipeline.
  Deliverable: one sentence with one number — "under realistic silicon constraints, accuracy
  is X% vs Y% in float32."
  Effort: 1 Puhti submit, no new code beyond what items 5–6 produced. ~1 day.

- [ ] **8. Lava simulation of the SNN.**
  Convert `SNNClassifier` via Lava-DL (Loihi 2's bit-accurate software simulator, no
  hardware needed). Verify simulated accuracy matches snnTorch baseline within 1 pp, report
  network resource usage (neurons, synapses, fan-in).
  Effort: ~100 lines for conversion + adapters. ~5 days.

- [ ] **9. Energy estimation from Loihi benchmarks.**
  Count synapse events per inference from the Lava simulation, multiply by Intel's published
  per-event energy figures, report estimated Loihi 2 energy per classification.
  Effort: ~1 day of bookkeeping over existing simulation outputs.

- [ ] **10. Cross-dataset generalization sweep.**
  Use the existing `run_puhti_dataset_test.sh` scaffolding to run on PhysionetMI, Cho2017,
  and BNCI2015_001. Even modest accuracy on the other datasets satisfies the
  "not BNCI2014_001-specific" reviewer concern.
  Effort: ~3 Puhti submits. ~1 week wall time.

- [ ] **11. Master results table.**
  One consolidated table covering: baseline → static6 → causal → Bessel → ADM →
  quantized end-to-end → cross-dataset. Per-subject columns, mean ± std. Each row supports
  a specific claim.
  Effort: ~1 day of Python plotting.

- [ ] **12. The mapping figure.**
  Production-quality version of the deck's mapping diagram, every primitive cited with a
  published silicon precedent (Mead 1989 for Gm-C, ReRAM papers for crossbar,
  Lichtsteiner/Liu for ADM, Davies et al. for Loihi).
  Effort: ~2 days in a vector tool (Inkscape/Affinity) or matplotlib.

- [ ] **13. Honest discussion section.**
  Name S2 / S5 limitations explicitly (cross-session non-stationarity is the bottleneck,
  not the classifier). Name the remaining gaps (no on-silicon measurement, training
  off-chip). Future-work hooks.
  Effort: writing — ~3 days as part of the manuscript.

- [ ] **14. Code release.**
  Tag the publication commit, freeze `requirements.txt` to exact versions, write a
  one-command reproduction script, add a README section "Reproducing the paper's results."
  Mint a Zenodo DOI.
  Effort: ~2 days.

- [ ] **15. Manuscript writing.**
  Abstract, introduction, methods, results, discussion. Stop-line: ~12-page IEEE format.
  Effort: ~3 weeks of focused writing.

- [ ] **16. Pre-submission checklist.**
  Reproduction tested from clean clone. Figures regenerable from scripts. Supplementary
  with raw per-fold CSVs. Code DOI in the README. Abstract leads with
  "fully neuromorphic-mappable."
  Effort: ~2 days.

---

## Priority 2 — Strengthens the paper, drop if calendar slips

Include only if items 1–16 finish ahead of schedule.

- [ ] **17. Apply for EBRAINS / SpiNNaker access.**
  Free, takes 2–4 weeks for approval. Apply this week regardless — having the option costs
  nothing. If approved in time, run a SpiNNaker measurement for cross-platform validation.
  Effort: 1 day for application, ~3 weeks if access granted and used.

- [ ] **18. Apply for Intel INRC (Loihi 2 access).**
  Same logic — apply now, use later if approved. Realistic approval timeline is 4–8 weeks.
  If approved post-submission, becomes a revision-stage strengthening.
  Effort: 1 day for application.

- [ ] **19. Reconstruction-fidelity validation for the ADM.**
  Beyond the simple RMSE check in item 2, run a more rigorous validation:
  ADM-encoded → reconstructed → re-classified, compare against direct classification.
  Strengthens the encoder claim.
  Effort: ~2 days.

- [ ] **20. Confidence calibration analysis.**
  Add normalized population scores to the WTA decoder, report mean confidence on correct vs
  incorrect predictions, compute expected calibration error. Reviewers like calibration
  evidence.
  Effort: ~10 lines in `evaluation.py` plus 1 day of plotting.

- [ ] **21. Windowed decoder ablation.**
  Re-evaluate using only spikes from t ∈ [500 ms, 2500 ms] to see if temporal weighting
  helps. Either result is publishable: "improves by X" or "decoder is robust to window
  choice."
  Effort: ~5 lines + 1 Puhti rerun.

- [ ] **22. Comparison against published neuromorphic EEG.**
  Situate against Corradi & Indiveri, Ceolini et al., recent FBCSP-SNN work. Mostly a
  literature-table addition to the discussion.
  Effort: ~3 days of reading + 1 day of writing.
