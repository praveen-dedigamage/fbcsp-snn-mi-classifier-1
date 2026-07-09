# FABLE5_AUDIT — Multi-Agent Publication-Readiness Audit

**Date:** 2026-07-07
**Scope:** FBCSP-SNN motor-imagery EEG classifier (IEEE TNSRE submission)
**Method:** Five independent read-only (Explore) audit agents formed conclusions from the
actual code (`fbcsp_snn/`, SLURM scripts) and actual paper text (`main.tex`,
`references.bib`) only. Their findings were then cross-referenced by the top-level agent
against four existing tracking documents:
`TODO.md`, `PAPER_REWRITE_NOTES.md`, `PIPELINE_REFERENCE.md`, `RESULTS_LOG.md`.

Paper path: `C:\Users\USER\Desktop\ieee_tnsre_paper\main.tex` (+ `references.bib`).
Code path: `D:\snn_pipeline_for_mi_eeg_classification\.claude\worktrees\hungry-neumann\`.

**Phase-2 status legend:** `CONFIRMS` (matches a tracked B-item / notes section),
`CONTRADICTS` (disagrees with something tracked — explained in place), `NEW` (not
previously tracked). Line numbers are as reported by the agents; a few differ by ±small
offsets between agents reading slightly different revisions — treat them as anchors.

---

## 1. What each agent reviewed, and how many findings

| Agent | Focus | Findings (raw) | Blocking | Important | Minor/other |
|-------|-------|:---:|:---:|:---:|:---:|
| **A** | Code-vs-paper equation & hyperparameter consistency | 8 | 2 (→ downgraded to minor after Phase 2) | 2 | 4 |
| **B** | Statistical & methodological rigor | 15 | 1 | 9 | 5 |
| **C** | Citation & reference integrity | 10 | 2 | 6 | 2 |
| **D** | Reproducibility & disclosure completeness | 15 | 4 (→ 3 after Phase 2) | 6 | 5 |
| **E** | Open-ended blind-spot search | 11 | 2 | 5 | 4 |

**Headline of the cross-reference:** the audit is largely *corroborating*. The great
majority of Phase-1 findings CONFIRM items already tracked as B1–B22 in `TODO.md`. Three
things stand out: (1) Agent A's two "blocking" items are downgraded to minor once the
SLURM overrides are accounted for; (2) two genuinely NEW and serious issues surfaced
that no tracking doc records — **no global PyTorch/NumPy seed is ever set**, and the
**paper internally contradicts itself on the BNCI2014-001 CV protocol**; (3) the latter
CONTRADICTS an explicit assertion in `PIPELINE_REFERENCE.md` that the paper's protocol
wording matches the code.

---

## 2. Findings by agent

### Agent A — Code-vs-paper consistency

**A1. `config.py:95-96` default `freq_bands` = 3 bands, paper (`main.tex:375-381`, Table
1) states 6 bands.** — Severity: **minor** (Agent A said blocking).
Phase-2: **CONTRADICTS Agent A's own severity / CONFIRMS what tracking establishes.**
`PIPELINE_REFERENCE.md` §7 and §2 confirm `run_puhti_array.sh` explicitly passes the fixed
6-band `--freq-bands`, so every reported result used 6 bands — the paper is correct and
the 3-band dataclass default is never exercised. This is stale-default hygiene, not a
results-affecting mismatch.

**A2. `csp_components_per_band` dataclass default = 8 vs argparse default = 4 vs paper
`m=4`.** — Severity: **minor** (Agent A said blocking, "internal inconsistency").
Phase-2: **CONTRADICTS Agent A's reading; CONFIRMS Agent D + tracking.** `8` is the
*total* components per band; `m = 8//2 = 4` per end (dual-end), which matches the paper's
`m=4` and its `2m×6×6 = 288` feature count (Agent D got this right; `PIPELINE_REFERENCE.md`
§7 confirms the script passes `--csp-components-per-band 8`). The only real issue is the
dataclass-vs-argparse default divergence (8 vs 4) — code hygiene, does not touch published
numbers.

**A17 / A(feature). `config.py:138` `feature_percentile=50.0` vs paper `main.tex:415-418`
"top 10%".** — Severity: **important.**
Phase-2: **CONFIRMS B10 — but tracking is deeper.** B10 (`TODO.md`) establishes the real
issue is worse than a 50-vs-10 number: `run_puhti_array.sh:111` passes `--mi-fraction 0.1`,
which selects by **threshold** (`MI ≥ 0.1 × max_MI`) and *overrides* `feature_percentile`
entirely — a *different algorithm* from the paper's "top 10% by rank," and the "≈57
features" estimate assumed percentile selection. Agents A/B/E all caught the number but
missed the algorithm-class mismatch that B10 documents.

**A20. `preprocessing.py:102` uses causal `sosfilt`; paper `main.tex:380-381` says
"zero-phase Butterworth (`sosfiltfilt`)".** — Severity: **important.**
Phase-2: **CONFIRMS the design fact / partially NEW for the paper text.** The causal-filter
design is well documented in code and was corrected in `CLAUDE.md` (`PIPELINE_REFERENCE.md`
§2a; it is also Pillar 1 of `PAPER_REWRITE_NOTES.md`). However, no tracked B-item flags
that **`main.tex:380` still literally says "zero-phase / sosfiltfilt"** — the paper text
fix is effectively untracked and must be made.

**A7 / A(T_s). Paper `main.tex:423` states `T_s=50` timesteps; code has no such constant
(tensor length = data-dependent).** — Severity: **important** (Agent A: unverifiable).
Phase-2: **CONFIRMS B22.** B22 establishes the real spike tensor is `(1001, …)` (raw EEG
sample count), so `T_s=50` is a stale/incorrect number that also throws off `eq:energy` by
~20×. `n_timesteps` is now saved to `pipeline_params.json` going forward.

**A18. MIBIF k-NN `k=3` (paper `main.tex:416`) relies on sklearn default, not pinned in
code.** — Severity: **minor.** Phase-2: **CONFIRMS** `PIPELINE_REFERENCE.md` §8b (fragility,
sklearn default happens to be 3 but is not pinned — one-line fix `n_neighbors=3`).

**A19 / A13. Van Rossum `tau_vr=10.0` used in code, not stated in paper.** — Severity:
**minor.** Phase-2: **CONFIRMS B13** (see D-side below; note tracking says it has since been
promoted to a `Config` field and saved to `pipeline_params.json`, but Table I still omits it).

**A28. `config.py:89` `n_folds=10` default vs paper's 5-fold framing.** — Severity:
**minor.** Phase-2: **CONFIRMS D-side** — the actual runs use 5 folds (`RESULTS_LOG.md`),
config default 10 is not what ran. Ties into the CV-protocol wording issue (see B1/D1 NEW
item below).

---

### Agent B — Statistical & methodological rigor

**B-stat-1. CV protocol is internally contradictory in the paper:** `main.tex:544` "session
1 train, session 2 test (no CV; fixed split)" vs `main.tex:676` "mean ± SD across 5 CV runs"
vs `main.tex:805` "66.2% under 5-fold CV". — Severity: **blocking.**
Phase-2: **NEW — and CONTRADICTS a tracking-doc claim.** No B-item records this internal
contradiction. Moreover `PIPELINE_REFERENCE.md:42` explicitly asserts the split logic
"matches what the paper's Datasets subsection and table captions say" — the audit shows it
does **not**: the paper says both things in different places. *Resolution I believe is
correct:* the runs do use 5 train/val folds of session 1, each evaluated against the
held-out session-2 test set (`val_fraction=0.2`, `RESULTS_LOG.md` "9 subjects × 5 folds"),
so the numbers are defensible — but the paper's prose must be made consistent (state
"5 train/val folds on session 1, fixed session-2 test set") rather than alternately calling
it "no CV" and "5-fold CV."

**B-stat-3 / B6-echo. Headline "outperforms SVM" is not statistically significant**
(paired t p≈0.0638, Wilcoxon p≈0.0547, n=9). — Severity: **important.**
Phase-2: **CONFIRMS B6** (already tested and documented, with the exact same p-values, in
`TODO.md` B6 and `PAPER_REWRITE_NOTES.md` §1/§2). Agent B independently reproduced B6.

**B-stat-7. n=9 subjects inadequate for a paired t-test; Wilcoxon preferred; effect sizes
absent.** — Severity: **important.** Phase-2: **CONFIRMS B6** (B6 already ran Wilcoxon
alongside the t-test). Adds the effect-size/Cohen's-d recommendation, which is **NEW** as an
explicit ask.

**B-stat-8 (cherry-pick). Abstract leads with the favorable BNCI2014-001 number and omits
that SVM beats the SNN 14/14 on Schirrmeister2017.** — Severity: **important.**
Phase-2: **CONFIRMS B7.**

**B-stat-2. INT8 degradation understated:** paper claims "mean FP32→INT8 difference 0.03%
(max 0.4%)" but the table shows mean |Δ|≈0.23 pp, max 0.70 pp (S6). — Severity:
**important.** Phase-2: **NEW.** Not tracked anywhere; a checkable numeric claim that
contradicts the paper's own table.

**B-stat-6. "INT8 Δ < 0.5%" bound is violated by individual subjects (0.7 pp observed).**
— Severity: **important.** Phase-2: **NEW** (same family as B-stat-2).

**B-stat-5. Literature-comparison arithmetic: "within 2.1 pp of BO-RF (68.1%)" but
68.1−66.2 = 1.9 pp.** — Severity: **minor.** Phase-2: **NEW** — and note the comparator
(BO-RF / `borf2021`) is itself a placeholder citation (see C2), so the base number is
unverified regardless.

**B-stat-13. No multiple-comparisons correction across the ~6-10 pairwise tests.** —
Severity: **minor.** Phase-2: **NEW.** (B6 flagged "decide whether to use a corrected
threshold" as a remaining sub-task, so partially anticipated; the explicit correction
recommendation is new.)

**B-stat-14. Contradictory table captions re: CV protocol** (`:676` "5 CV runs" vs `:715`
"across folds"). — Severity: **important.** Phase-2: **NEW** (ties to B-stat-1).

**B-stat-9. Energy equation's 0.15 spikes/neuron/timestep is the input-encoder rate, not a
whole-network measurement.** — Severity: **important.** Phase-2: **CONFIRMS B8** (problem 2)
and **B15**. Tracking shows this is the single most-worked item: event-counting
instrumentation (`mean_events_per_trial`) now measures the real rate, code built but not yet
run on Puhti.

**B-stat-11. Feature selection 10% (paper) vs 50% (code default).** — Severity: **important.**
Phase-2: **CONFIRMS B10** (see A17 note — real issue is threshold-vs-percentile mode).

**Accurate / honest (B-stat-4, -8-Schirr, -12):** SNN-vs-LDA p<0.005 verified correct; "SVM
> SNN on all 14" verified correct; Schirrmeister loss reported honestly. Phase-2:
**CONFIRMS B6** (the LDA-significant / SVM-loss framing is exactly the honest story
`PAPER_REWRITE_NOTES.md` §2 prescribes). No action beyond keeping it.

**B-stat-10, -15. Report median/IQR for high-variance subjects; report Cohen's d.** —
Severity: **minor.** Phase-2: **NEW** (good-practice additions).

---

### Agent C — Citation & reference integrity

**C1. `tfanet2022` is a placeholder** — `author = {…and others}`, `title = {TODO: Verify…}`,
`journal = {TODO…}`, yet cited as "TFANet 84.9%" (`main.tex:235,570,662`). — Severity:
**blocking.** Phase-2: **CONFIRMS B1.**

**C2. `borf2021` is a placeholder** — `author = {TODO}`, `title = {TODO…}`, cited as
"BO-RF 68.1%" (`main.tex:573,664`). — Severity: **blocking.** Phase-2: **CONFIRMS B1.**
(Also the base number behind B-stat-5's arithmetic error.)

**C9. `wang2020fbcsp_snn` carries an in-file `TODO: Verify this is the correct … paper`**
(`main.tex:160,246`). — Severity: **important.** Phase-2: **CONFIRMS B1.**

**C10. `virgilio2020snn_mi` carries `TODO: Verify journal/page details`** (`main.tex:162,248`).
— Severity: **important.** Phase-2: **CONFIRMS B1.**

**C3. `sun2022eeg_snn` orphaned placeholder stub** (not cited). — Severity: **minor.**
Phase-2: **CONFIRMS B1** (B1 explicitly says "delete the orphaned unused stub
`sun2022eeg_snn`").

**C6. `eshraghian2021snntorch` key says 2021 but `year = {2023}`.** — Severity: **important.**
Phase-2: **NEW.** Not tracked. (Attribution is likely valid — Proc. IEEE 2023 — but the
metadata is inconsistent.)

**C7. `bnci2015001` named "2015" but `year = {2012}` (Faller et al.).** — Severity:
**important.** Phase-2: **NEW.** (Nomenclatural, not a fabrication.)

**C8. `kostas2020bendr` lists "Bhatt, Juliusz" twice (duplicate author).** — Severity:
**important.** Phase-2: **NEW.** Sloppy metadata; suggests uncurated entry.

**C4/C5. `song2022transformer_eeg`, `welch1967psd` orphaned (present, not cited).** —
Severity: **minor.** Phase-2: **NEW** (harmless, curation hygiene).

*Verified-plausible landmark cites* (ramoser2000csp, ang2008fbcsp, davies2018loihi,
neftci2019surrogate, vanrossum2001metric, moabb2018, bnci2014001, etc.): no action.

---

### Agent D — Reproducibility & disclosure completeness

**D-seed. No global `torch.manual_seed()` / `np.random.seed()` / `random.seed()` anywhere;
only sklearn splitters get `random_state=42`.** — Severity: **blocking.**
Phase-2: **NEW — most important reproducibility finding.** No tracking doc records this.
SNN weight init, dropout, and any torch-side RNG vary run-to-run, so results are not bit-
reproducible and the reliability/Monte-Carlo work built on "seed reproducibility" rests on
per-call seeds only, not a global seed. (Note: CLAUDE.md's "Deterministic splits" claim
refers only to the sklearn splits, which is technically true but oversells reproducibility.)

**D1. Paper says BNCI2014-001 = session split, but SLURM runs 5 folds.** — Severity:
**blocking → reclassify as the same issue as B-stat-1 (paper-wording, not a code bug).**
Phase-2: **NEW / CONTRADICTS `PIPELINE_REFERENCE.md:42`** (same as B-stat-1). The code is
internally coherent (5 train/val folds × fixed session-2 test); the paper's wording is not.

**D-feat. `config.py` default `feature_percentile=50.0` vs SLURM `--mi-fraction 0.1`.** —
Severity: **blocking → important.** Phase-2: **CONFIRMS B10.** Downgrade rationale: the
*paper* reports the 10%/threshold value that actually ran; the discrepancy is a config-
default trap for re-runners, not a wrong published number. Still important (fix the default
or document the override).

**D-filter. `sosfilt` (causal) vs paper "zero-phase sosfiltfilt".** — Severity: **blocking →
important.** Phase-2: **CONFIRMS** A20 and the design fact (`PIPELINE_REFERENCE.md` §2a).
The paper-text correction at `main.tex:380` remains to be made.

**D-EA. Euclidean Alignment (`euclidean_alignment=True`) enabled by default, applied every
run, but absent from Table 1 and the CSP Methods subsection.** — Severity: **important.**
Phase-2: **CONFIRMS B9** (which pairs it with the Riemannian-mean issue below).

**D-riemannian (via B9). Paper Eq. `cov` says "arithmetic mean of per-trial covariances";
code defaults to `riemannian_mean=True` (Fréchet mean).** — Severity: **important.**
Phase-2: **CONFIRMS B9.** (Agent D noted EA; B9/`PIPELINE_REFERENCE.md` §5a adds that the
covariance-averaging equation itself describes the wrong method — script audit confirms
Riemannian mean actually ran.)

**D-slope. `model.py:84` `fast_sigmoid(slope=25)` hardcoded, not in paper/Table 1.** —
Severity: **important.** Phase-2: **CONFIRMS B12.**

**D-tau / D-batch. `tau_vr=10.0` and `batch_size=64` used every fold, absent from Table 1.**
— Severity: **important.** Phase-2: **CONFIRMS B13** (tracking notes both were promoted to
`Config` fields; Table 1 still omits them).

**D-valsplit. `val_fraction=0.2` internal train/val split not clearly disclosed for
BNCI2014-001.** — Severity: **important.** Phase-2: **NEW** (component of the CV-wording
cluster; not separately tracked).

**D-misc-minor:** Bessel filter option undisclosed (**NEW**, minor — `PIPELINE_REFERENCE.md`
§2a notes Bessel was tried at 63.4% and set aside); `n_folds=10` default vs 5 run
(**CONFIRMS** A28); AMP undisclosed (**NEW**, minor); MI `random_state=42` seed undisclosed
(**NEW**, minor).

*Correctly disclosed (no action):* software versions (`main.tex:582`), hardware V100/Puhti
(`:585`), 250 Hz, 9 subjects, epoch window 0.5–4.0 s, notch, baseline correction — all match.

---

### Agent E — Open-ended blind-spot search

**E1. Filter causality claim contradicts implementation** (`main.tex:380` "zero-phase
sosfiltfilt" vs causal `sosfilt`), undermining the paper's own streaming/neuromorphic
premise. — Severity: **blocking.** Phase-2: **CONFIRMS** A20/D-filter; the paper-text fix is
untracked.

**E2. Feature-selection 10% vs 50% changes the reported SNN input dimensionality / energy.**
— Severity: **blocking.** Phase-2: **CONFIRMS B10.**

**E3. Internal feature-count contradiction:** `main.tex:406-407` "…= 288" vs `main.tex:413`
"All 576 features…". — Severity: **important.** Phase-2: **partially tracked / mostly NEW.**
B22 mentions "the 576-vs-288 feature count" in passing as another stale-number instance, but
there is no dedicated fix item; the contradiction stands in the text.

**E4/E5/E6. Visible `\TODO{}` macros remain in the submitted manuscript:** pending Loihi 2 /
Lava energy figures in the abstract (`:86,96`), empty binary-classification results table
(`:524,528`), and unwritten author biographies (`:748,754`). — Severity: **important.**
Phase-2: **CONFIRMS B2 (Table V), B3 (bios), B8 (energy figures).** Tracking shows the code
to produce these numbers exists but has not been run on Puhti (`RESULTS_LOG.md` — all runs
"not yet submitted"/"awaiting completion").

**E7. Conflicting primary accuracy: README "65.8%" (hardware-compatible) vs `main.tex:636`
"66.2%".** — Severity: **important.** Phase-2: **NEW.** Not tracked; needs one canonical
headline number.

**E8. Paper fixes Butterworth but code was also evaluated with Bessel (63.4%, undisclosed).**
— Severity: **minor.** Phase-2: **CONFIRMS** D-misc (Bessel), consistent with
`PIPELINE_REFERENCE.md` §2a.

**E9. `eq:energy` uses published Loihi specs + an input-only spike rate, presented in the
analysis.** — Severity: **minor.** Phase-2: **CONFIRMS B8** (all three B8 problems: SNN-only
scope, wrong firing rate, no baseline). Tracking shows a substantially reworked
`compute_energy.py` (real Lava SynOps + cited GPU/EEGNet-M4 baseline) exists but is not yet
run.

**E10. "All 576 features" prose ambiguous about z-norm/selection ordering.** — Severity:
**minor.** Phase-2: **NEW** (minor; code ordering is correct).

**E11. Limitations says "no systematic ablation… performed," but README lists many; and
`main.tex:214` still claims "adaptive band selection" (removed from the pipeline).** —
Severity: **minor.** Phase-2: **CONFIRMS B11** (ablation code built, not yet run) and **B7**
(the stale "adaptive band selection" line is explicitly called out in B7).

---

## 3. What's still needed to publish — prioritized punch list

Ordered by priority, synthesizing all five agents + the four trackers. Items marked
**[NEW]** were surfaced by this audit and are not in any tracking doc; **[tracked: Bn]**
already have an entry.

### Tier 0 — Research-integrity blockers (must fix; cheap, text-only)
1. **Replace or remove the two placeholder citations with fabricated numbers** —
   `tfanet2022` (84.9%) and `borf2021` (68.1%), plus resolve the `wang2020fbcsp_snn` and
   `virgilio2020snn_mi` "verify" TODOs and delete the `sun2022eeg_snn` stub.
   *[tracked: B1; Agent C1–C3,C9,C10]* Citing accuracy numbers to `TODO`-body references is
   the single most likely integrity flag for a reviewer.
2. **Fix the filter-causality claim** at `main.tex:380` — change "zero-phase Butterworth
   (`sosfiltfilt`)" to the causal single-pass `sosfilt` the code actually uses, and turn
   that into a *strength* (streaming/neuromorphic compatibility). *[Agent A20/D/E1; the
   design is tracked, but the paper-text error is effectively **[NEW]** — no fix item
   exists.]*
3. **Reconcile the CV-protocol contradiction** — the paper alternately says "no CV; fixed
   split" (`:544`) and "5-fold CV / 5 CV runs" (`:676,:805,:715`). State the real protocol
   once: 5 train/val folds on session 1, evaluated on the fixed session-2 test set.
   **[NEW]** — and note this *contradicts* `PIPELINE_REFERENCE.md:42`'s claim that the paper
   already matches the code.

### Tier 1 — Reproducibility & correctness blockers
4. **Set global RNG seeds** (`torch.manual_seed`, `np.random.seed`, `random.seed`) at
   program entry, and disclose the seed in the paper. Without this the FP32 numbers are not
   reproducible run-to-run. **[NEW]** — highest-value untracked finding.
5. **Fix or document the feature-selection description.** Paper says "top 10% by rank"; code
   ran `--mi-fraction 0.1` = threshold mode (`MI ≥ 0.1·max_MI`), overriding the 50% config
   default. Rewrite `main.tex:415-418` to the actual algorithm and re-derive the "≈57
   features" figure. *[tracked: B10; Agents A17/B-stat-11/D/E2]*
6. **Correct the INT8-degradation claims.** "mean 0.03% / max 0.4%" and "Δ<0.5%" are
   contradicted by the paper's own table (mean ≈0.23 pp, max 0.70 pp on S6). **[NEW]**
7. **Resolve the `T_s=50` vs ~1001-timestep error** and its ~20× effect on `eq:energy`.
   *[tracked: B22; Agents A7/E]*
8. **Fix the internal feature-count contradiction** (288 vs 576, `main.tex:406-407` vs
   `:413`). Mostly **[NEW]** (only mentioned in passing under B22).

### Tier 2 — Disclosure gaps (important; needed for a complete Methods/Table 1)
9. **Add undisclosed hyperparameters to Table 1:** surrogate slope=25 *(B12)*,
   `tau_vr=10.0` and `batch_size=64` *(B13)*, and disclose the `val_fraction=0.2` split
   *(partly NEW)*.
10. **Document Euclidean Alignment and correct the covariance-averaging equation** (paper
    says arithmetic mean; code ran Riemannian/Fréchet mean, EA whitening undocumented).
    *[tracked: B9; Agent D]*
11. **Clean up citation metadata:** `eshraghian2021snntorch` year (2021 key vs 2023 field),
    `bnci2015001` naming (2015 vs 2012), `kostas2020bendr` duplicate author, remove orphaned
    entries. All **[NEW]** (Agent C6–C8, C4/C5).
12. **Reconcile the headline accuracy number** (README 65.8% vs paper 66.2%) to one
    canonical value. **[NEW]** (Agent E7).

### Tier 3 — Framing & honesty (important; reshapes abstract/conclusion)
13. **Rewrite the abstract/conclusion to the honest, significance-tested story:** SNN beats
    LDA (p<0.005, both datasets), ties SVM on BNCI2014-001 (p≈0.05–0.06, *not* significant,
    n=9), loses to SVM on Schirrmeister2017 (14/14). Add the stats subsection, report
    Wilcoxon + effect sizes, and decide on a multiple-comparison stance. Remove the stale
    "adaptive band selection" line (`main.tex:214`) and the unqualified "orders of magnitude
    smaller than deep CNN baselines" claim (contradicts the EEGNet <3,000-param cite, B14).
    *[tracked: B6/B7/B14; Agents B-stat-3,7,8,13,15]*

### Tier 4 — Missing results (blocked on HPC runs, code already built)
14. **Run and fill the pending experiments**, then remove all `\TODO{}` macros:
    binary-classification Table V (Cho2017 + BNCI2015-001, *B2*), the three ablations that
    isolate the abstract's three claimed contributions (*B11*), the reliability/noise sweep
    and measured `mean_events_per_trial` that properly back the energy claim (*B8/B15*), and
    the Lava-measured Loihi 2 energy figure with its cited GPU baseline. Author biographies
    (*B3*). All code exists per `RESULTS_LOG.md`; none has been run — this is the "submit →
    wait → copy" loop.

### Tier 5 — Nice-to-have / hygiene (non-blocking)
15. Pin `n_neighbors=3` in MIBIF (sklearn default, not pinned — *B8b*); report median/IQR
    for high-variance subjects; disclose or drop the Bessel/AMP options; delete the stale
    `run_puhti_subject.sh`; decide whether Fig. 1 should show the baseline branch and the
    quantisation experiments it currently omits.

---

## 4. Cross-reference summary (counts)

- **CONFIRMS existing tracking:** the bulk — B1 (C1–C3,C9,C10), B6 (B-stat-3,7,8;
  B-stat-4/8/12 honest-framing), B7, B8 (B-stat-9, E9), B9 (D-EA, D-riemannian), B10
  (A17, B-stat-11, D-feat, E2), B11 (E11), B12 (D-slope), B13 (A19, D-tau/batch), B14
  (framing), B15 (B-stat-9), B22 (A7, E).
- **NEW (untracked):** no global torch/numpy seed **(most significant)**; paper-internal
  CV-protocol contradiction; INT8-degradation numbers understated / "<0.5%" violated;
  BO-RF 2.1-vs-1.9 arithmetic; multiple-comparisons absence; several citation-metadata
  defects (eshraghian year, bnci2015 naming, kostas duplicate author, orphaned entries);
  README-vs-paper headline number; `val_fraction` disclosure; the untracked *paper-text*
  correction of `main.tex:380`.
- **CONTRADICTS tracking:** `PIPELINE_REFERENCE.md:42` states the paper's split protocol
  "matches what the paper's Datasets subsection and table captions say" — the audit shows
  the paper contradicts *itself* on this, so that assertion is wrong. Separately, Agent A's
  two self-declared "blocking" items (3-vs-6 bands; csp_components 8-vs-4) are downgraded to
  minor once the SLURM overrides and the total-vs-per-end distinction are accounted for.
```
