# Session checkpoint — 2026-07-10, written ahead of context compaction

Everything below is already committed/pushed or saved to disk — this file
is just an index so nothing gets lost or re-derived incorrectly after
compaction. Full detail for each item lives in the files referenced, not
duplicated here.

## Code repo — committed and pushed, HEAD = `233a83c`

In order, this session (see `git log --oneline` for the full list):
1. `caac3b8` — fixed `inject_beta_noise` CUDA/CPU device mismatch.
2. `da05174` — redesigned `bandpass_filter_noisy` to perturb cutoff
   frequencies (stable by construction) instead of raw `sos` coefficients,
   which was destabilising IIR filter poles and collapsing accuracy to
   exact chance. Bumped `run_puhti_reliability.sh`'s `--time` 30min→2.5h.
3. `0ea946f` — same fix pattern applied to `inject_ea_whitener_noise`
   (perturb the whitener's own eigenvalues, not raw matrix entries) — found
   because it showed the identical failure signature as #2.
4. `f1a5f44`, `60dd349`, `5074704`, `4480653`, `53617a1` — reliability sweep
   validated (job `35412677`, all 9 subjects × 5 folds, zero errors),
   aggregated (`aggregate_reliability.py`), plotted
   (`plot_reliability_sweep.py`, 3×3 small-multiples in **pipeline order**:
   Filter Bank→EA→CSP→Z-norm→Encoder→SNN→Joint), written into the paper.
5. `0dd3714`, `233a83c` — energy computation completed (Lava/SLAYER
   simulation + cross-check against this codebase's own B15 instrumentation),
   written into the paper, raw terminal output logged.

**Full narrative and exact numbers**: `RESULTS_LOG.md` (this repo).
**Raw data, independent of any summary**: `puhti_logs/Results_verify/`
(7 files — analyze summary, joint quantisation sweep, band-selection
root-cause trace, significance test rerun, reliability sweep raw + energy
computation raw).
**Code-level technical detail**: `PIPELINE_REFERENCE.md` §11a/§11c (the two
noise-injection redesigns), §12 (energy script extension).

## Paper — saved to disk (not a git repo, no push needed)

`C:\Users\USER\Desktop\ieee_tnsre_paper\main.tex` compiles clean (10 pages,
only one pre-existing cosmetic `ieeecolor.cls`/xcolor warning, zero
undefined references). This session's edits:
- New "Hardware-Realism Reliability Sweep" Results subsection
  (`sec:reliability`) + Figure (`fig:reliability`,
  `figures/reliability_sweep.pdf`).
- "Neuromorphic Deployment Implications" subsection (`sec:energy`)
  rewritten around the real Lava-measured figures, replacing the old
  placeholder `eq:energy` entirely — new Table VII (`tab:energy`).
- Six new citations in `references.bib` (five confirmed via
  `compute_energy.py`'s own comments; `burrello2020eegnet_m4` lower
  confidence — flagged in `TODO.md` B1).
- Table I's "Ranges (Hz)" overflow fixed (added spaces after commas so the
  `p{3.0cm}` column can actually wrap — it has no other break points).
- B8, B15, B22, B9, B10, B24 (new) all marked done in `TODO.md` with full
  write-ups. `PAPER_REWRITE_NOTES.md` §3/§4 updated to match.

**Full write-up**: `TODO.md` + `PAPER_REWRITE_NOTES.md`, both in the paper
folder.

## Open items — genuinely unresolved, not just undocumented

1. **Schirrmeister2017 retrain — status UNCONFIRMED.** Command given
   (`bash submit_schirrmeister.sh Results_schirrmeister_verify`) but the
   user has not yet pasted back `squeue` output or confirmed it was
   actually submitted. Table IV's numbers (58.8/47.5/64.8%) remain stale
   (adaptive-band pipeline, same root cause as B23) until this runs.
   Compatibility was checked (not just assumed) — no confirmed blocker,
   but Riemannian mean convergence at 128×128 (vs. the 22×22 case already
   validated) is genuininely untested under the current codebase. Read the
   actual logs carefully when this comes back, don't assume success.
2. **Evaluation-protocol description may not match the code — not yet
   investigated.** `main.tex`'s "Evaluation Protocol" subsection states
   *"For single-session datasets (Schirrmeister2017, Cho2017,
   BNCI2015-001), 5-fold stratified cross-validation is applied on the
   full session."* But `datasets.py::_stratified_split` (and
   `submit_schirrmeister.sh`'s own comment) describes a single 80/20
   `StratifiedShuffleSplit`, not 5-fold CV. These could both be true if the
   80/20 split fixes a held-out test set once and 5-fold CV then only
   applies to train/val *within* the 80% — but this has not been traced
   through the code to confirm. Worth checking before or alongside the
   Schirrmeister2017 results coming back.
3. **Figure 1 redesign — waiting on the user.** User wants to combine the
   existing TikZ block diagram with a signal-transformation visualisation
   (what the EEG signal actually looks like at each stage), not a separate
   standalone figure. They're sketching their vision and will share it with
   more specific direction. A first draft (5-panel signal gallery, NOT the
   integrated design they actually want) exists at
   `figure_drafts/signal_flow_demo.py` / `.png` in this worktree — moved
   there from the ephemeral session scratchpad so it survives across
   sessions, but **deliberately left uncommitted to git** (not authorised
   for the paper or the repo yet). Superseded once the user's sketch
   arrives — treat as reference only, not a starting point to iterate on
   without seeing their actual design first.

## Remaining experiment list (unchanged from before today)

Ablations (single-end CSP, fixed encoder, cross-entropy loss), Cho2017,
BNCI2015-001 — none yet submitted. Recommended order and submit commands:
`RESULTS_LOG.md`'s "How to submit each experiment" section.
