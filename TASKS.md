# Task board

↑ [Project README](README.md) · [STATUS](STATUS.md) · [INDEX (registry + ledger)](INDEX.md) · [Rules (AGENTS.md)](AGENTS.md)

**Read this before starting any work.** It is the single place that says what is being worked on right now and
what is waiting. Finished work lives in the [INDEX registry](INDEX.md#3-experiment-registry); claims and their status live
in the [hypothesis ledger](INDEX.md#4-hypothesis-ledger).

## How to claim a task (all agents and humans)

1. **Pull first** (`git pull`), then check **In progress** below. Do not take a task someone else holds.
2. **Claim before working:** pick a row whose State is **Ready** (and a free CPU slot if it is heavy), move it from
   Backlog to In progress, fill in *Owner* (session ID or name), *Started* (date), *Folder* and *CPU*, then **commit and push that edit alone** before doing anything else. If the push is
   rejected because someone else pushed, pull and re-check; if they claimed it first, pick another task.
3. **Work only in your folder** (rules in [AGENTS.md](AGENTS.md)). Update *Notes* if the plan changes materially.
4. **Finish:** remove the row from In progress, add your experiment to the [INDEX registry](INDEX.md#3-experiment-registry)
   (or ask the DRI to), and list any new tasks you discovered in Backlog.
5. **Stale claims:** a claim with no commit in its folder for 24 hours may be taken over. Note the takeover in *Notes*.
6. **Add new ideas to the Backlog as soon as you have them**: new approaches to try, follow-ups, needed checks and bugs
   in shared code. Use the next free ID (T-numbers are never reused) and give each a goal, a "done when" and pointers,
   even if you will not do it yourself. If you start it right away, reserve it in the same commit.
7. **Releasing:** if you stop or are interrupted, move your task back to Backlog with a note on how far you got.

## In progress

**CPU slots:** the machine has 4 shared cores. At most **2 heavy tasks** (long model runs, training, sweeps) may be
In progress at once; light tasks (analysis, docs, small runs under ~10 CPU-minutes) are unlimited. Count the `heavy`
rows below before claiming a heavy task; if both slots are taken, take a light task or wait.

| Task | Owner | Started | Folder | CPU | Notes |
|---|---|---|---|---|---|
| T8: **Causal/orthogonal test of H-M3**; check the H-M1 phase clock across other common gate sets | Codex sub-agent mechanism_checks (session 216ca1055f0c81918ec742d3f3a2161f) | 2026-10-16 | [m16/approaches/mechanism_checks/](m16/approaches/mechanism_checks/) | light | Firms up or withdraws remaining mechanism claims. Pointers: [m16/FINDINGS.md](m16/FINDINGS.md), [m16/modes.py](m16/modes.py). Isolated Git worktree; single-threaded checks under 10 CPU-minutes; parent coordinates board updates and pushes. |
| T7: **Finish M_32,10 (E5)** and **sweep the label constant budget** (1, 3, 5, 10, 20 constants) | Codex sub-agent algzoo_labels (session 216ca1055f0c81918ec742d3f3a2161f) | 2026-10-16 | [algzoo/constant_budget/](algzoo/constant_budget/) | light | Tests whether "legible but not understood" scales with size or is a constant-budget artifact. Pointers: [algzoo](algzoo/README.md), ledger H-A1. Isolated Git worktree; single-threaded checks under 10 CPU-minutes; parent coordinates board updates and pushes. |
| T19: **Crossover compute per method** (headline metric): for each mechanistic estimator, the largest budget at which it still matches sampling (MSE ratio <= 1), plus a cheap mechanistic family that wins at low budgets (100-4k FPE) and is pushed upward | Codex (session 216ca1055f0c81918ec742d3f3a2161f) | 2026-10-16 | [m16/approaches/crossover_compute/](m16/approaches/crossover_compute/) | light | ARC asks for MSE vs compute, not one budget; a mechanistic method that wins only below some compute C* is partial progress measured by C*. Done when the v3 report prints C* per method and at least one mechanistic estimator has C* > 0 on all groups, or none can. Pointers: [m16/leaderboard.py](m16/leaderboard.py) (`--budgets curve`), T15. Standalone report and 11 failure-mode tests implemented; independent review and frozen cheap Gaussian candidate evaluation in progress. Shared leaderboard integration stays DRI-only; runs remain light. |
| T4: **Re-score leaderboard v2 original-model rows** with the pooled 28M truth | DRI (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/leaderboard.py](m16/leaderboard.py) (shared code: DRI-only) | light | v2's orig rows used the old truth (~0.025 pp low). Done when the orig rows in [results/leaderboard.json](results/leaderboard.json) are refreshed. Plan: superseded by the v3 re-run (truths read at report time, so rescoring is automatic) |
| T17: **Harsher noise levels** (sigma 0.01-0.02 x RMS) in the leaderboard, reported per level; more truth samples for large budgets | DRI (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/leaderboard.py](m16/leaderboard.py) (shared code: DRI-only) | heavy | "Significantly alter accuracy" may mean more noise than our 0.0025/0.005 |
| T15: **Score the full MSE-vs-compute curve** (~1k to 1M evals) and report where each method crosses sampling | DRI (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/leaderboard.py](m16/leaderboard.py) (shared code: DRI-only) | heavy | ARC asks to match sampling across compute; v2 tested only 16k/64k (tube won at 64k and lost at 256k) |
| T1: **Stochastic reduction of the tie margin** (deductive): carry the low inputs as a distribution and track the exact law of the residual tie margin given the top values/positions | DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/approaches/stoch_margin/](m16/approaches/stoch_margin/) | heavy | Unblocked by DRI: leaderboard v3 (T14/T16 code) is pushed; development may start, final scoring after the T14/T16 audit. The one untried deductive direction (suggested by E23 and E27). Done when an estimator is scored on [leaderboard v2](m16/leaderboard.py) with a verdict |
| T2: **Replicate the H x T scan with 3+ training seeds per (H,T)** (the 'why' part moved to T10) | DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/approaches/ht_seeds/](m16/approaches/ht_seeds/) | heavy | Uses the zoo's pre-trained seeds 0-4 per (H,T) with sci_tao/scan_HT.py's method (copied, not edited). The ladder plan rests on H-E11, which has one training seed per cell and wide CIs. Done when the collapse point per H is confirmed or revised with CIs |

## Backlog (highest priority first)

**You may claim only rows whose State is Ready.** Take the highest Ready row whose CPU class fits the free slots.
*Blocked* rows name what they wait for; when you finish a task, change the State of every row it unblocks to Ready
(in the same commit that removes your row from In progress). Shared code ([m16/leaderboard.py](m16/leaderboard.py),
[m16/arc_eval.py](m16/arc_eval.py), [m16/common.py](m16/common.py)) is DRI-only: if your task needs a change there,
add a Backlog row describing it instead of editing.

Priorities revised 2026-10-17: (1) make the leaderboard measure ARC's challenge correctly (DRI, in progress);
(2) verify the premise of the ladder (H-E11 has one seed per cell) before building on it; (3) the ladder (T10 -> T11)
with the independent stochastic attack (T1) in parallel; (4) expensive measurement before any 'solved' claim;
(5) everything not on the ARC critical path (T3, T6-T8 are Ready now so idle agents have work, but rank below T2).

| ID | State | Task | Why / done when | Pointers |
|---|---|---|---|---|
| T20 | **Ready** (light) | **Make each estimator's cost honest and portable**: implement the incremental (prefix-reuse) passes that tube, sci_laplace, sci_tao, sci_feynman, boundary and control_variate charge for, add `ADVICE`/`SUPPORTS`/`PRECOMPUTE_FLOPS` declarations, re-tune any orig-fitted constants on the DEV models | The v3 audit found charged cost 1.8-2.3x below the implemented work and undeclared orig tuning; leaderboard OVERRIDES currently rescale costs. Done when each estimator's measured implemented work equals its reported cost (instrumented count as in review/estimator_audit/fullpass_count.py) and the OVERRIDES entry can be removed. Work in a NEW folder per estimator (e.g. `m16/approaches/<name>_v2/`), do not edit the original | [review/estimator_audit/README.md](review/estimator_audit/README.md), [m16/leaderboard.py](m16/leaderboard.py) OVERRIDES |
| T3 | **Ready** (heavy) | **Symbols from rotating-neuron gate switches** instead of records (Poincare's untried variant) | Ceiling test like E27: done when the reduced system's exact bias is measured on all 18 models | [sci_poincare](m16/approaches/sci_poincare/README.md) |
| T6 | **Ready** (heavy; needs LLM agents) | **Qwen H2 re-run**: >=100 controls (top neurons projected out, or dense rotation), 2-3 explainers per item, n-gram dedupe for judges | Resolves the "inconclusive" verdict on the essay's claim | [RESULTS.md](RESULTS.md), [review](review/README.md), ledger H-Q1..Q4 |
| T10 | Blocked: T2 | **Minimal pair: H=16, T=4 (calculation wins 38-266x) vs T=5 (7.7x)**: find exactly which error patches appear at T=5 that do not exist at T=4, then fix the calculation for T=5. Absorbs T2's 'why does it become heavy-tailed' and T13's diagnostics | Smallest instance of the open problem (simplification ladder). Done when a deductive estimator beats sampling at H=16, T=5 on the leaderboard, or the new patches are characterised precisely | [sci_tao](m16/approaches/sci_tao/README.md) (`scan_HT.json`), ledger H-E11/H-E12; `zoo_2nd_argmax(16, T)` |
| T11 | Blocked: T10 | **Climb the ladder**: once T10 works, step T=5->6->...->10 at H=16 and check it survives H=8/32; each step must beat sampling on the leaderboard before moving on | Turns a small-model solution into the M_16,10 solution, or shows exactly where it breaks | [m16/leaderboard.py](m16/leaderboard.py) (generalise to (H,T)) |
| T12 | Blocked: T10 | **Training-stage dimension**: track when the hard error patches form across checkpoints (`n_seqs` = 2^20..2^30) for H=16, T=5 and T=10 | Shows which learned feature creates the heavy-tailed bands | `zoo_2nd_argmax(16, T, n_seqs=...)` |
| T9 | Blocked: ARC sharing their method; T14-T17 | **Score ARC's method** on our harness when it is shared | The comparison the project was set up for. Needs ARC's estimator in `est(model, budget, rng)` form, or their numbers | [m16/arc_eval.py](m16/arc_eval.py), [m16/leaderboard.py](m16/leaderboard.py), ledger H-E4/H-E5 |
| T5 | Blocked: user approval for more compute | **Brute-force exact reference** (needs much more compute): first test whether exact 10-D cone masses can be computed fast and accurately on a sample of regions | Gives an exact answer to compare with ARC; benchmark only. Go/no-go after the cone-mass test | INDEX open questions; [cone_enum](m16/approaches/cone_enum/README.md); ledger H-M7 |
| T13 | Closed (merged into T10) | *(merged into T10: its error-family / positive-only diagnostics are tools for the minimal pair)* | - | - |

## Recently finished (details in the INDEX registry)

T21 (stratified sampler: one pooled rejection stream, candidates charged, ~1.1 FPE per sample), T14 + T16 (leaderboard v3 classes and compute rule, with DRI rulings on the [estimator audit](review/estimator_audit/README.md): cost as implemented, advice flags, rejection charged), T18 (leaderboard for any (H,T): `--H --T`; truths for H=16, T=4 and T=5 in `results/leaderboard_truths_H16_T{4,5}.json`), E10-E27, LB2 (leaderboard v2), R1 (review): see [INDEX section 3](INDEX.md#3-experiment-registry) and the
[approaches index](m16/approaches/README.md).
