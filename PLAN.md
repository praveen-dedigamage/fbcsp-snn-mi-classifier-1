# Publication Plan — FBCSP-SNN Neuromorphic EEG Classifier

Generated with Opus 4.6. Work through items in order; Priority 1 items are non-negotiable
for a publishable paper.

---

## Priority 1 — Required for a publishable paper

Without these, the paper either can't be written or won't survive review.

- [ ] **1. Bessel-filter experiment.**
  Already coded (commit 1631610), already on the queue. Submit to Puhti when BUs return.
  Deliverable: one row in the master results table; addresses the S7 group-delay regression.
  Effort: 1 day of work + ~12 hours of Puhti wall time.

- [ ] **2. Adaptive ADM encoder.**
  Replace `|x[t] − x[t−1]|` with reference-tracking `|x − v_ref|` plus ON/OFF polarity.
  Validate locally on one subject with reconstruction RMSE < 5%.
  Effort: ~30 lines of code in `encoding.py`, plus pipeline plumbing for doubled feature
  dimension. ~1 day.

- [ ] **3. Persistent-state flag.**
  Hoist filter `zi`, encoder `v_ref`, and LIF membrane potential `V` out of per-trial scope
  so they persist across trial boundaries. Required for the streaming claim and for any future
  Loihi deployment.
  Effort: ~30 lines across `encoding.py` and `pipeline.py`. ~1 day.

- [ ] **4. ADM A/B sweep across all 9 subjects, 5 folds.**
  Frame as parity check (Δmean ≥ −1.0 pp is success, not failure).
  Effort: 1 Puhti submit, no new code. ~12 hours wall time.

- [ ] **5. CSP weight quantization sweep.**
  Quantize CSP eigenvectors to 4-, 6-, 8-bit symmetric per-tensor. Run all 9 subjects at
  each level. Mirror the existing INT8 SNN methodology.
  Deliverable: accuracy-vs-precision cliff plot. Target: ≤1 pp drop at 6-bit.
  Effort: ~50 lines for the quantization wrapper, then 3 Puhti submits. ~3 days.

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
