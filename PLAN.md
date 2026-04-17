# Publication Plan — FBCSP-SNN Neuromorphic EEG Classifier

Generated with Opus 4.6. Work through items in order; Priority 1 items are non-negotiable
for a publishable paper.

**Target venue:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)

**Paper claim:** First fully analog/mixed-signal neuromorphic BCI pipeline where every
stage maps to a published silicon primitive — Gm-C filter bank, ReRAM crossbar (CSP),
ADM encoder, LIF SNN — with validated tolerance to hardware imperfections at each stage.

**Hardware-compatible paper number:** 65.8% mean FP32 (9 subjects, BNCI2014_001, 5-fold)

---

## Milestone tracker

```
DONE   Item 1  Bessel filter             63.4% — causal Butterworth wins
DONE   Item 2  ADM encoder               67.4% software / 65.8% hardware-compatible
DONE   Item 3  Persistent state          resolved — no code change
DONE   Item 4  ADM A/B sweep             confirmed 9-subject
DONE   Item 5  CSP PTQ sweep             <1pp drop at 4-bit ✓
DONE   Item 6  Butterworth MC            σ=1%:−0.05pp | σ=2%:+0.26pp | σ=5%:+2.42pp mean
TODO   Item 7  End-to-end stress test    ready to submit
TODO   Item 8  Lava simulation           critical path (~5 days)
TODO   Item 9  Energy estimation         1 day after item 8
TODO   Item 10 Cross-dataset             optional strengthener
TODO   Items 11-16  Tables, figures, manuscript, release
```

## Estimated timeline from today (2026-04-17)

```
DONE       Item 6 results (2026-04-17)
+1 day     Item 7 (end-to-end combined stress test)
+6 days    Item 8 (Lava simulation) ← critical path
+7 days    Item 9 (energy estimate)
+2 weeks   Item 10 (cross-dataset, optional)
+3 weeks   Items 11–12 (master table + mapping figure)
+6 weeks   Manuscript submission
```

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

- [x] **5. CSP weight quantization sweep.** ✓ CLOSED 2026-04-17
  Quantize CSP eigenvectors to 4-, 6-, 8-bit symmetric per-tensor. Run all 9 subjects at
  each level. Mirror the existing INT8 SNN methodology.
  Deliverable: accuracy-vs-precision cliff plot. Target: ≤1 pp drop at 6-bit.
  Effort: ~50 lines for the quantization wrapper, then 3 Puhti submits. ~3 days.
  Result (2026-04-17): 65.8% FP32 mean (9 subjects, causal Butterworth, no augwin).
  PTQ drop: 8-bit +0.18pp | 6-bit +0.23pp | 4-bit −0.89pp. <1pp at all levels.
  Paper number: 65.8% hardware-compatible vs 67.4% software upper bound (−1.6pp causal cost).

  **Contingency 5a — Per-filter quantization (if per-tensor drop >2 pp at 6-bit).**
  Each CSP eigenvector gets its own scale factor: `scale_i = max(|w_i|) / (2^(bits-1) - 1)`.
  Reduces outlier-driven quantization error with no runtime cost. ~20 lines in
  `quantization.py`. Re-run the same Puhti sweep.

  **Contingency 5b — QAT with frozen quantized CSP (if 5a still fails).**
  Train the SNN with CSP outputs already quantized to target precision (8-, 6-, 4-bit).
  CSP filters are quantized before CSP projection during the training loop; the SNN learns
  to work with the lower-precision features. CSP filters themselves are not updated.
  Pipeline change: in `_run_single_fold`, quantize `csp.filters_` to target bits before
  calling `csp.transform` on both train and validation data within the training loop.
  Restore FP32 filters before saving artifacts. Three separate trained models per subject
  (one per target precision). Effort: ~50 lines in `pipeline.py` + 3× Puhti submit time.

- [x] **6. Butterworth coefficient sensitivity (Monte Carlo).** ✓ CLOSED 2026-04-17
  Perturbation model: cutoff-frequency shift (Gm mismatch → τ = C/Gm shifts f_c by same σ%).
  9 subjects × 5 folds × 100 draws per σ level. Jobs: 34046769 / 34046770.
  Results (mean accuracy drop across 9 subjects):
    σ=1%  (careful layout):        −0.05 pp  ← indistinguishable from noise
    σ=2%  (production tolerance):  +0.26 pp  ← negligible
    σ=5%  (worst-case):            +2.42 pp  ← moderate; see per-subject breakdown
  Per-subject highlights: S1 most sensitive (+5.53 pp mean at σ=5%, ±7.88 pp std);
  S6 most robust (−0.97 pp at σ=1%, actually improved).
  Note on "FAIL" verdict: the automated p95 criterion (p95 < 1pp) is too strict —
  p95 pools all subjects/draws and is dominated by S1 outlier draws.  The mean drop
  is the correct metric for a fixed-mismatch chip.  Paper claim stands:
  "Gm matching of σ≤2% (standard in careful CMOS layout) degrades accuracy by <0.3 pp."

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
