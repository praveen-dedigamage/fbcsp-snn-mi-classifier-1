# Results Log — Puhti Submission Cycles

Tracks the "submit → wait → copy" cycle for every experiment implemented on
2026-07-06 (see `PIPELINE_REFERENCE.md` §10 for what was built, `TODO.md`
B2/B11/B15/Tier-0 in the paper folder for why). Paste Puhti output (terminal
summaries, `summary.csv` contents, `reliability_results.json` contents) into
the matching section below as each run completes. Keep unresolved runs at
the top of their section; move completed ones to "Done" once numbers are
final and copied into the paper.

---

## Schirrmeister2017 retrain — PARTIAL, 5 of 14 subjects need resubmit (2026-07-10)

`bash submit_schirrmeister.sh Results_schirrmeister_verify` (job `35414792`
training array, 70 tasks = 14 subjects × 5 folds; `35414793`–`35414806`
per-subject aggregates; `35414807` analyze).

**Result**: 8 of 70 training tasks hit `TIMEOUT` at exactly `04:00:17` —
the 4-hour wall-time budget, not a code crash or OOM kill (confirmed via
`sacct -j 35414792 --format=JobID,State,ExitCode,Elapsed,MaxRSS -X`, which
shows `State=TIMEOUT` explicitly, distinct from `FAILED`/`OUT_OF_MEMORY`).
Mapped task IDs back to (subject, fold) via `task = (S-1)*5 + fold + 1`:

| Subject | Timed-out folds | Aggregate (`sacct`) |
|---|---|---|
| S2  | fold 2 (1/5)          | CANCELLED |
| S3  | fold 3 (1/5)          | CANCELLED |
| S4  | folds 0,1,2,4 (4/5)   | CANCELLED |
| S7  | fold 2 (1/5)          | CANCELLED |
| S12 | fold 3 (1/5)          | CANCELLED |

The other 9 subjects (S1, S5, S6, S8, S9, S10, S11, S13, S14) completed all
5 folds cleanly — `sacct` shows `COMPLETED, ExitCode=0:0` throughout, no
resubmit needed for those. Because each subject's aggregate job depends
(`afterok`) on all 5 of its own fold tasks, one timeout was enough to
cancel that subject's aggregate — which cascaded to cancel the final
analyze job (`35414807`) too, since *it* depends on all 14 aggregates.

Worth noting: even the tasks that *did* complete ranged up to `03:44:26`
elapsed — several came within ~15 minutes of the 4-hour limit, so this
wasn't just one pathological subject (though S4 losing 4/5 folds stands
out). Consistent with the risk flagged before this run: Riemannian mean
convergence at 128×128 (vs. the 22×22 case already validated on
BNCI2014-001) is genuinely slower for a meaningful fraction of
subject/fold combinations, not just S4.

**Fixed**: `submit_schirrmeister.sh`'s `SBATCH_TIME` doubled from `4:00:00`
to `8:00:00` (was hardcoded via `export`, so an external env var override
wouldn't have worked without editing the script itself).

**Resubmit — only the 5 affected subjects, no need to redo the other 9**:
```bash
git pull
SUBJECTS="2 3 4 7 12" bash submit_schirrmeister.sh Results_schirrmeister_verify
```
`submit_schirrmeister.sh` respects a pre-set `SUBJECTS` env var and skips
its own default `seq 1 14` generation, so this correctly scopes to just
these 5 (confirmed by reading the script, not assumed). Note this will
harmlessly re-run all 5 folds for each of these subjects (including the
ones that already completed, e.g. S2's folds 0,1,3,4) since
`submit_puhti.sh` builds per-subject fold arrays as a whole block, not at
individual-fold granularity — wasted compute is small relative to the
whole run, not worth the fragility of trying to cherry-pick individual
folds instead.

**Important — the automatic analyze step from this resubmission will only
cover these 5 subjects**, since `submit_puhti.sh`'s dependency chain is
built from the `SUBJECTS` list passed to it. Once this resubmission's
aggregates all show `COMPLETED`, trigger the real, full 14-subject analyze
manually:
```bash
RESULTS_DIR=Results_schirrmeister_verify SUBJECTS="1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
    sbatch run_puhti_analyze.sh
```

---

## Verification retrain — COMPLETE 2026-07-08 (all 9 subjects)

`bash submit_puhti.sh Results_verify` — 9 subjects × 5 folds, code at commit
`0ca8101` (fully pulled and confirmed before submission).

- Train job: `35390599` (45 tasks)
- Aggregate jobs (one per subject, each depends only on its own 5 fold tasks):
  `35390600`–`35390608` (subjects 1–9 respectively)
- Analyze job: `35390609` (depends on all 9 aggregates) — final summary lands
  in `logs/fbcsp_analyze_35390609.out`

**What to check once complete:**
1. `Results_verify/Subject_N/summary.csv`'s `test_acc_fp32`/`test_acc_int8`/
   `test_acc_csp_{8,6,4}bit` vs. the original `Results/` — should match
   (regression check, see "What verification actually checks" section below).
2. `val_acc_lda`/`svm` and `test_acc_lda`/`svm` — expected to differ on
   purpose (Tier 0 tuning).
3. New fields now present: `mean_events_per_trial`,
   `mean_input_events_per_trial`, `mean_hidden_events_per_trial`,
   `n_timesteps` — no "before" to compare, this is new evidence.
4. New joint-PTQ fields: `test_acc_joint_csp{8,6,4}_snn{8,6,4}` (9 combos) —
   also new evidence.

**Outcome, not what was expected going in**: this was NOT a same-pipeline
regression check. The original `Results/` was generated with a 12-band
adaptive Fisher-selection front end (`band_selection.py`), which this
session's own earlier commit `a9ceddf` had already deleted in favour of the
fixed six-band scheme, without saying so in the commit message. Full
diagnostic trail and corrected significance-test results (SNN now
*significantly* beats SVM, no longer significantly beats LDA — a flip from
before) are in `puhti_logs/Results_verify/` (`BAND_SELECTION_FINDING.md`,
`ANALYSIS.md`, `SIGNIFICANCE_TEST_RERUN.md`). `main.tex`, `TODO.md`, and
`PAPER_REWRITE_NOTES.md` (paper folder) have all been updated with the real
numbers. Schirrmeister2017 needs the same retrain — not done yet.

Unblocked now: `sbatch run_puhti_reliability.sh` (needs these checkpoints)
and `python compute_energy.py --results-dir Results_verify ...` (needs the
event-count fields, confirmed present in `Results_verify/Subject_1/
summary.csv`).

---

## Reliability sweep — COMPLETE AND VALIDATED 2026-07-09 (job 35412677)

Fourth attempt (job `35412677`) ran clean: 45/45 `reliability_results.json`
files, zero tracebacks. Sanity-checked `Subject_1/fold_0` against all 9
sweeps — every known bug signature (instant collapse to exact chance,
zero variance, exploding event counts) is gone:

| Sweep | severity 0→0.3 | Verdict |
|---|---|---|
| `csp_weight_noise` | 0.823→0.370 | graceful |
| `snn_weight_noise` | 0.823→0.638 | graceful, most robust to weight noise |
| `beta_noise` | 0.823→0.701 | graceful, most robust overall |
| `filter_bank_noise` | 0.823→0.574 | graceful (§11a fix confirmed) |
| `ea_whitener_noise` | 0.823→0.746 | graceful (§11c fix confirmed — was 0.823→0.250 before) |
| `znorm_noise` | 0.823→0.356 | graceful (steeper late, no instant collapse) |
| `encoder_threshold_noise` | 0.823→0.821 | nearly flat — plausible real finding, not a bug (see below) |
| `encoder_adaptation_noise` | 0.823→0.359 | graceful, genuinely sensitive |
| `joint_noise_all_sources` | 0.823→0.250 | now a believable *gradual* convergence to chance (was instant before) |

**`encoder_threshold_noise`'s flatness is plausibly genuine, not a bug**:
the encoder's threshold evolves via `decay=0.95` every timestep across
~1001 timesteps, so the *initial* threshold value's influence washes out
almost completely (`0.95^100 ≈ 0.006`) long before the sequence ends —
worth a sentence in the paper write-up rather than treating as suspicious.

**This experiment is done.** Only spot-checked Subject_1/fold_0 in detail;
the other 44 files exist and have no tracebacks, but haven't been
individually eyeballed — reasonable to trust given both known failure modes
are demonstrably fixed and no other subject/fold reported an error.

**Aggregated and written into the paper 2026-07-09**: `aggregate_reliability.py`
(cross-subject mean±SD across all 9 subjects, matching Table III's
convention — full command: `python aggregate_reliability.py --results-dir
Results_verify`). Plotted via `plot_reliability_sweep.py` as a 3×3
small-multiples figure (2D chosen over a single overlaid plot — 9 sources
with very different sensitivity profiles would bury the flat ones under the
steep ones — and over a 3D surface, which is hard to read precisely in a
static print figure). Output copied to the paper repo's `figures/
reliability_sweep.pdf`, wired into `main.tex`'s new "Hardware-Realism
Reliability Sweep" Results subsection. Abstract/Introduction `\TODO{}`
markers resolved. See `PAPER_REWRITE_NOTES.md` §4 (paper folder) for the
full write-up and key findings.

### Third attempt (job 35411451, 2026-07-09) — completed all 45 tasks, confirmed the filter fix, found one more bug

Ran clean: `find ... | wc -l` → 45, zero tracebacks in any `.err`. Sanity-checked
`Subject_1/fold_0/reliability_results.json`:

- **`filter_bank_noise` fix confirmed working**: now shows a genuine
  graceful decline (0.823→0.766→0.691→0.613→0.574 across severities
  0→0.05→0.1→0.2→0.3), not an instant collapse. Good.
- **But `ea_whitener_noise` showed the exact same failure signature the
  filter bank had**: collapse to near-chance almost immediately
  (0.823→0.282→0.269→0.250→0.250) *and* its event count exploding ~4x
  (39417→154610). Root cause confirmed to be the same underlying design
  flaw as §11a, applied to the EA whitener matrix instead of filter
  coefficients — see `PIPELINE_REFERENCE.md` §11c for the full trace.
  **Fixed**: `inject_ea_whitener_noise` now perturbs the whitener's own
  eigenvalues (guaranteed symmetric positive-definite reconstruction) rather
  than its raw matrix entries. Stress-tested: 200 draws × 4 severities,
  every draw stayed positive-definite, worst-case peak-magnitude growth
  only ~1.43× at the highest severity.
- **`joint_noise_all_sources` was confounded by the same bug** (already
  near-chance by severity 0.05) — not a genuine joint-fragility finding on
  its own, since one of its 7 combined sources was broken. Will look
  different once re-run with the EA fix.
- Everything else looked plausible: `csp_weight_noise`, `snn_weight_noise`,
  `beta_noise`, `znorm_noise`, `encoder_adaptation_noise` all show graceful,
  non-zero-variance declines. `encoder_threshold_noise` is nearly flat
  (0.823→0.821 across all severities) — plausibly genuine, not obviously a
  bug: the encoder's threshold evolves via `decay=0.95` every timestep over
  ~1001 timesteps, so the *initial* threshold value's influence washes out
  almost completely (`0.95^100 ≈ 0.006`) long before the sequence ends —
  worth keeping in mind when writing this up, but not treated as suspicious
  the way the exact-chance/zero-variance pattern was.

**Consequence: none of job 35411451's `ea_whitener_noise` or
`joint_noise_all_sources` results should be used** (the other 7 sweeps'
results for this job are fine and don't need to be discarded, but simplest
to just do one more full resubmit rather than patching together sweeps from
two different runs).

```
RESULTS_DIR=Results_verify sbatch run_puhti_reliability.sh
```
Once complete, verify with:
```
find Results_verify -name "reliability_results.json" | wc -l   # expect 45
grep -l "Traceback (most recent call last)" logs/fbcsp_rel_*.err   # expect empty (scope to the new job's ID range)
cat Results_verify/Subject_1/fold_0/reliability_results.json   # check ea_whitener_noise + filter_bank_noise both show graceful declines now
```

### Second attempt (job 35408852, 2026-07-09) — right directory, still failed, 2 more problems found

Correctly targeted `RESULTS_DIR=Results_verify` this time (fix from the
first attempt worked), but failed for two new, unrelated reasons:

1. **`--time=00:30:00` too short.** In 30 minutes, only got through
   `csp_weight_noise`, `snn_weight_noise`, `beta_noise`, and 3/5 severities
   of `filter_bank_noise` before `slurmstepd` killed it: `CANCELLED ... DUE
   TO TIME LIMIT`. `joint_noise_all_sources` (all 7 sources combined,
   likely the most expensive sweep) never started. **Fixed**: bumped to
   `--time=02:30:00` in `run_puhti_reliability.sh`, with a comment
   explaining why.
2. **Real bug in `filter_bank_noise`, not a hardware-realism finding.**
   Every severity collapsed to exactly chance level (0.2500 ± 0.0000, zero
   variance across 20 repeats) starting at the mildest severity (5%), with
   `RuntimeWarning: overflow encountered in cast` in the same log. Traced
   to noise being added directly to the `sos` filter's pole-determining
   coefficients (`a1`/`a2`) at a scale that reliably pushed poles outside
   the unit circle for narrowband sections (empirically confirmed: one
   section's pole margin was only `0.0214`, versus a noise std of `0.10` at
   just 5% severity) — an unstable filter's output diverges, producing
   garbage the SNN can't classify, which is exactly the deterministic
   collapse observed. **User-directed fix**: redesigned
   `bandpass_filter_noisy()` to perturb the filter's *design parameters*
   (cutoff frequencies) before synthesis instead of the discretised
   coefficients after — stable by construction (`butter`/`bessel` always
   return a stable filter for valid cutoffs), and arguably the more
   physically accurate model of component tolerance besides. Stress-tested
   2000 draws across all bands/severities: zero unstable filters. Full
   detail: `PIPELINE_REFERENCE.md` §11a.

**Consequence: none of job 35408852's `reliability_results.json` output
should be used** — it used the buggy noise model and covered only 4/9
sweeps. Superseded by the third attempt above (job 35411451), which then
found the EA whitener bug — see there for the current resubmit instructions.

### First attempt (FAILED 2026-07-08, 2 bugs, both fixed) — for the record

`sbatch run_puhti_reliability.sh` was run without setting `RESULTS_DIR`
first. Two problems surfaced:

1. **Wrong directory, not a code bug.** The script defaults to
   `RESULTS_DIR="${RESULTS_DIR:-Results}"` — plain `Results`, not
   `Results_verify`. The job's own startup banner confirmed `Results:
   Results`. Worse: `Results/Subject_1/fold_0`'s log showed `Filter bank: 3
   bands` — a *third*, even older pipeline generation (the original
   3-non-overlapping-band static scheme), different from both the 12-band
   adaptive pipeline traced in `BAND_SELECTION_FINDING.md` (found via
   Subject 8) and the current fixed six-band pipeline. `Results/` is an
   inconsistent grab-bag across subjects from different points in the
   project's history, not a single coherent snapshot — never use it as a
   reference again; always pass `RESULTS_DIR=Results_verify` explicitly.

2. **Real bug, now fixed**: `inject_beta_noise` (`fbcsp_snn/quantization.py`)
   crashed every task on the `beta_noise` sweep with `RuntimeError: Expected
   all tensors to be on the same device, but found at least two devices,
   cuda:0 and cpu!`. Root cause: `torch.Generator()` (no `device=` arg) and
   `torch.randn(n_neurons, generator=gen)` (no `device=` arg) always
   allocate on CPU, while `lif.beta` lives on the model's device (`cuda:0`
   on Puhti). This function was written without local CUDA access
   (documented in its own docstring as unverified) — first real GPU run
   caught it immediately. Every sibling noise function was audited for the
   same risk: `inject_csp_filter_noise`/`inject_ea_whitener_noise`/
   `inject_znorm_noise` are pure numpy (no device concept, safe);
   `inject_model_weight_noise` and the encoder threshold/adaptation noise
   functions already correctly derive `device=` from the actual tensor at
   every call site (established pattern in `inject_weight_noise_tensor`).
   `inject_beta_noise` was the only offender. Fixed by generating the noise
   tensor on CPU (unchanged RNG semantics/seeding) then moving it to
   `nominal.device` before combining.

**Next**: commit + push the fix, `git pull` on Puhti, then resubmit with
the directory set explicitly this time:
```bash
RESULTS_DIR=Results_verify sbatch run_puhti_reliability.sh
```

---

## Energy computation — COMPLETE 2026-07-10, real measured figures in the paper

Discovered mid-session that Puhti already had a real, working
`compute_energy.py` + `run_lava_infer.py` (never git-tracked, so invisible
to this repo) — considerably more developed than assumed: uses **real
measured Lava SynOps** (not an assumption), has **real cited references**
for an analog front-end (Qian 2017, Verhoeven 2007, Sharifshazileh 2021,
Burr 2017), and a **real cited GPU/EEGNet-M4 comparison** (Burrello et al.
2020, 4.28 mJ measured). Decision: extend it, don't replace it.

**Important clarification from the user:** Loihi 2 is a **digital**,
asynchronous, event-driven chip — not analog. The existing script's Gm-C/
ADM/ReRAM front-end estimates describe a *separate*, more speculative
hardware story (a hypothetical all-analog front-end that could pair with a
digital Loihi backend), not a claim that Loihi itself is analog. Now stated
explicitly in the script's docstring and print output so this doesn't get
conflated later.

**What got added:** a cross-check section using B15's `mean_input_events_
per_trial`/`mean_hidden_events_per_trial` (this codebase's own PyTorch-side
spike instrumentation) to compute an *independent* fan-out-weighted SynOps
estimate, reported side-by-side against Lava's measured figure per subject.
Rationale: Lava's number stays primary (it's the real target framework's
own accounting for real, currently-deployable hardware — stronger than
anything derived from PyTorch instrumentation alone), but if the two
independent measurements roughly agree, that's corroborating evidence
before either number is trusted in the paper; if they diverge, that's worth
investigating first. Verified the cross-check's core arithmetic
(`input_events × n_hidden + hidden_events × n_output`) and its graceful
skip-on-missing-data behavior with mock `pipeline_params.json` fixtures
before considering this done.

**Correction to the "never git-tracked" note above**: `run_lava_infer.py`
and `save_test_spikes.py` turned out to actually be git-tracked (added
commit `00b8e1a`, well before this session) — the "invisible to this repo"
framing referred to an even older untracked copy on Puhti at the time this
was first investigated. Confirmed 2026-07-10 when the same two files were
found physically missing from the Puhti working tree (`mv`'d into an
archive folder outside git's oversight during an earlier cleanup) —
restored cleanly via `git checkout HEAD -- save_test_spikes.py
run_lava_infer.py`, no data lost, since git had them all along.

**Full run, 2026-07-10, all 9 subjects × 5 folds, zero errors:**
```bash
python save_test_spikes.py --results-dir Results_verify --subjects 1 2 3 4 5 6 7 8 9 --n-folds 5
source .venv_lava/bin/activate   # turned out unneeded — main .venv already had lava-nc/lava-dl
python run_lava_infer.py --results-dir Results_verify --subjects 1 2 3 4 5 6 7 8 9 --n-folds 5 --output-dir Results_lava
python compute_energy.py --lava-dir Results_lava --results-dir Results_verify --subjects 1 2 3 4 5 6 7 8 9 --n-folds 5
```

**Results:**
- Lava/SLAYER accuracy vs. trained FP32: 66.0% → 65.6%, mean gap −0.40pp
  (well under the script's own 1pp tolerance) — one fold (S7 fold 2) showed
  a +18.75pp single-fold outlier, but S7's subject-level aggregate
  (+3.68pp) is unremarkable; not investigated further given the tight
  overall mean.
- Measured mean SynOps/trial: 1,741,693 (input firing 6.1%, hidden 19.4%).
- Cross-check (Lava-measured vs. this codebase's B15 PyTorch-side
  estimate): mean ratio 1.14, tightly clustered 1.10–1.19 across all 9
  subjects — consistent systematic offset, not scattered disagreement,
  corroborating both measurements.
- Loihi 2: 13.9 µJ/inference (measured, ~8pJ/SynOp). Comparison table added
  to the paper (Loihi 1, Edge CPU, GPU V100 at two utilisation levels, all
  non-Loihi figures explicitly back-of-envelope).

Written into `main.tex`'s "Neuromorphic Deployment Implications" subsection
(new label `sec:energy`) as `tab:energy`, completely replacing the old
`eq:energy` placeholder rather than just correcting its arithmetic.
Abstract and Introduction `\TODO{}` markers resolved. Six new citations
added to the paper's `references.bib` (see `TODO.md` B1/B8, paper folder,
for verification status). Full write-up: `PAPER_REWRITE_NOTES.md` §3.
and check the cross-check table for agreement before trusting either the
Lava-measured or PyTorch-derived SynOps figure in the paper.

**Also decided:** start fresh on Puhti — dozens of old `Results_*`
directories and loose scripts (`run_butterworth_mc.py`, `run_e2e_stress.py`,
`run_lava_infer.py`, `save_test_spikes.py`, `show_csp_quantized.py`, and the
old untracked `compute_energy.py`) get archived (moved, not deleted) by the
user before the next pull — the new tracked `compute_energy.py` would
otherwise conflict with the untracked one of the same name on `git pull`.

---

## Decided run order (2026-07-07)

1. **Verification retrain — `Results_verify` (new dir, do NOT overwrite `Results/`).**
   `bash submit_puhti.sh Results_verify` — no ablation flags, default full
   pipeline. Purpose: confirm the new codebase reproduces the original
   paper-producing numbers before trusting anything built on top of it.
   Original `Results/` must stay untouched as the reference — retraining
   into the same paths would silently overwrite the checkpoints being
   verified against, with no way back.
2. **Reliability sweep** — `sbatch run_puhti_reliability.sh`, once step 1
   confirms no regression (or immediately against untouched `Results/` if
   those checkpoints are still there — the sweep needs no retraining either
   way, it's pure inference on a saved checkpoint).
3. **B2 — Cho2017 + BNCI2015-001** (fills empty Table V).
4. **B11 — the three ablations** (essential evidence for the abstract's
   three claimed contributions).

### What "verification" actually checks — isolated vs. joint vs. new fields

Not everything in `summary.csv` is checked the same way:

- **Isolated fields** (`test_acc_fp32`, `test_acc_int8`,
  `test_acc_csp_{8,6,4}bit`) → **regression check.** These are the only
  fields with a "before" to compare against (they're exactly what's already
  in the original `Results/` and the paper's Tables IV/VI). Should match.
  Kept in the code only for this comparison and because `analyze_results.py`
  reads these exact keys — **not** meant to be the primary hardware-realism
  evidence going forward. That's what the joint sweep is for (B15 Tier D:
  it's supposed to *replace* these scattered numbers in the paper, not sit
  alongside them as equally-weighted results).
- **Joint fields** (`test_acc_joint_csp{X}_snn{Y}`) → **new evidence, no
  prior number to check.** First time this grid has ever been computed —
  nothing tested "CSP=8bit and SNN=6bit simultaneously" before this session.
- **`mean_events_per_trial`** → also brand new, no prior baseline. This is
  the number that should replace B8's unmeasured "0.15 spikes/neuron/
  timestep" assumption once it comes back from Puhti.
- **`val_acc_lda`/`test_acc_lda`/`val_acc_svm`/`test_acc_svm`** → **expected
  to differ from the original run, on purpose.** Tier 0 tuning intentionally
  changed how these are computed (LDA shrinkage, SVM grid search) — a
  mismatch here is the tuning working, not a regression.

---

## Scope gap in the reliability sweep — CLOSED 2026-07-07

Originally flagged: Tier B noise-tested CSP weights, SNN weights, and LIF
beta only — a reasonable minimum-viable set, but not sufficient to fully
back a blanket "this pipeline is analog-circuit-realizable" claim. **All
five gaps below are now implemented and wired into `run_reliability()`.**

1. **Filter bank coefficients (Butterworth/Bessel `sos` values) — DONE.**
   New `bandpass_filter_noisy`/`apply_filter_bank_noisy` in
   `preprocessing.py`, wired as the `filter_bank_noise` sweep. This was the
   most important gap: the paper's hardware-realizability argument
   specifically leans on the bandpass filter being Gm-C-circuit realizable
   (§2a), and that claim had never been stress-tested against actual analog
   component-value variation until now.
   - **Real bug caught and fixed during implementation:** naively adding
     noise to the full `sos` array breaks scipy's requirement that column 3
     (the `a0` coefficient) stay exactly `1.0` — `sosfilt` rejects the array
     otherwise. Fixed by only perturbing the other 5 columns per section
     (also the physically correct choice: `a0=1` is a normalisation
     convention, not itself a component value). Verified with both
     Butterworth and Bessel filters before considering this done.
2. **Spike encoder's `adapt_inc`/`decay` — DONE.** Generalized the noisy
   encoder kernel (`_adaptive_threshold_encode_noisy_jit`) to accept
   per-feature `adapt_inc`/`decay` tensors, not just per-feature threshold.
   Wired as two sweeps: `encoder_threshold_noise` (comparator offset —
   this was actually built in the original Tier B pass but never wired into
   a runnable sweep, caught and fixed now too) and `encoder_adaptation_noise`
   (adapt_inc + decay together).
3. **Z-normalisation mean/std — DONE.** New `inject_znorm_noise` in
   `quantization.py`, wired as the `znorm_noise` sweep.
4. **Euclidean Alignment whitener matrix — DONE.** New
   `inject_ea_whitener_noise` in `quantization.py` (mirrors
   `inject_csp_filter_noise`'s structure exactly, same dict-of-matrices
   shape), wired as the `ea_whitener_noise` sweep. Degrades gracefully to a
   no-op if a fold's CSP was fit with `euclidean_alignment=False`.
5. **SNN biases — DONE.** `inject_model_weight_noise` now perturbs
   `Linear.bias` by default (`include_bias=True`), alongside weights, in the
   same `snn_weight_noise` sweep.
6. **Still correctly out of scope, no gap:** MIBIF feature selection
   (routing/wire selection, not a continuous analog coefficient) and Van
   Rossum loss (training-only, no inference-time circuit).

**New experiment added on top of closing the gaps:** `joint_noise_all_sources`
— every one of the 7 individual noise sources perturbed simultaneously at
matched severity, since a real chip has every stage imperfect at once and
testing sources in isolation doesn't represent the actual deployment
scenario (same principle as `pipeline.py`'s joint CSP+SNN quantisation grid,
extended here to the full noise picture).

`run_reliability()` now runs **9 sweeps total**: `csp_weight_noise`,
`snn_weight_noise`, `beta_noise`, `filter_bank_noise`, `ea_whitener_noise`,
`znorm_noise`, `encoder_threshold_noise`, `encoder_adaptation_noise`,
`joint_noise_all_sources`. Full technical detail: `PIPELINE_REFERENCE.md`
§11.

---

## How to submit each experiment

```bash
cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
git pull                                   # pick up this session's changes
mkdir -p logs

# B2 — missing binary-classification datasets (Table V)
bash submit_cho2017.sh                     # -> Results_cho2017/
bash submit_bnci2015.sh                    # -> Results_bnci2015/  (script already existed)

# B11 — ablation study (one component at a time vs. full pipeline)
bash submit_ablation_single_end_csp.sh     # -> Results_ablation_single_end_csp/
bash submit_ablation_fixed_encoder.sh      # -> Results_ablation_fixed_encoder/
bash submit_ablation_cross_entropy.sh      # -> Results_ablation_cross_entropy/

# B15 — reliability / hardware-noise sweep (run AFTER the corresponding
# training + aggregate jobs have completed — needs saved checkpoints)
sbatch run_puhti_reliability.sh            # -> Results/Subject_*/fold_*/reliability_results.json

# Baseline Tier 0 tuning is automatic — already baked into every training
# run above (baseline.py), no separate submission needed. Just re-run any
# training experiment and the tuned SVM/LDA numbers come along with it.
```

Monitor with `squeue -u $USER`; per-subject summaries land in
`Results_*/Subject_N/summary.csv`, final cross-subject analysis in
`logs/fbcsp_analyze_<JOBID>.out`.

---

## 1. B2 — Missing binary-classification results (Table V)

**Status:** not yet submitted.

### Cho2017 (2-class, 52 subjects, 64ch, 512Hz)
*(paste `summary.csv` / analyze output here once complete)*

### BNCI2015-001 (2-class, 12 subjects, 13ch, 512Hz)
*(paste `summary.csv` / analyze output here once complete)*

---

## 2. B11 — Ablation study

**Status:** not yet submitted. Compare each against the full-pipeline
baseline (BNCI2014-001, `Results/`) already in the paper's Table III.

### Single-end CSP (`--csp-single-end`)
*(paste `summary.csv` here — compare mean accuracy vs. dual-end baseline)*

### Fixed-threshold encoder (`--encoder-type fixed`)
*(paste `summary.csv` here — compare vs. adaptive-threshold baseline)*

### Cross-entropy loss (`--loss-type cross_entropy`)
*(paste `summary.csv` here — compare vs. Van Rossum baseline)*

---

## 3. B15 — Reliability / hardware-realism sweep

**Status:** not yet submitted. Requires training to have completed first
(reads saved `best_model.pt`/`pipeline_params.json` per fold).

### Joint CSP+SNN quantisation grid
*(this is saved automatically during training as
`test_acc_joint_csp{8,6,4}_snn{8,6,4}` in `pipeline_params.json`/
`summary.csv` — no separate run needed; paste the grid here once training
completes)*

### Noise-robustness sweep (`reliability_results.json` per fold)
*(paste per-fold JSON or a summarised table here: severity → acc_mean ±
acc_std, events_mean ± events_std, for each of csp_weight_noise,
snn_weight_noise, beta_noise)*

### Measured spike-events-per-trial (fixes B8's energy estimate)
*(paste `mean_events_per_trial` from `summary.csv` here — this replaces the
paper's unmeasured "0.15 spikes/neuron/timestep" assumption)*

---

## 4. Baseline Tier 0 — Tuned SVM/LDA re-validation

**Status:** not yet re-run with tuning. Automatic on next training run of
any experiment (no separate submission).

### Re-run B6 significance tests against tuned baseline
*(once a fresh BNCI2014-001 + Schirrmeister2017 training run with tuned
baselines completes, re-run `sig_test.py`-style paired tests — see
`TODO.md` B6 in the paper folder for the original methodology — and paste
updated p-values here)*

---

## Done

*(move completed, paper-ready results here once copied into `main.tex`)*
