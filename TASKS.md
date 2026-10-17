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
| T29: **Tensor-rank feasibility of exact preintegrated error mass** | Codex sub-agent tensor_probe (session 216ca1055f0c81918ec742d3f3a2161f) | 2026-10-16 | [m16/approaches/tensor_mass/](m16/approaches/tensor_mass/) | light | New Ready task reserved immediately. Test DEV-only split-variable singular spectra and held-out cross approximation of exact tie-band mass. Done when a low-rank surrogate has error below matched sampling with charged construction cost, or bounded probes show why it fails. Numerical compression is unreviewed/quadrature unless a structural argument is established; no solved claim from determinism. Under 300 single-thread CPU seconds. |
| T20: **Make each estimator's cost honest and portable**: implement the incremental (prefix-reuse) passes that tube, sci_laplace, sci_tao, sci_feynman, boundary and control_variate charge for, add `ADVICE`/`SUPPORTS`/`PRECOMPUTE_FLOPS` declarations, re-tune any orig-fitted constants on the DEV models | Codex sub-agent incremental_walker (session 216ca1055f0c81918ec742d3f3a2161f) | 2026-10-16 | [m16/approaches/sci_tao_v2/](m16/approaches/sci_tao_v2/) | light | Start with an actually incremental exact segment walker and portable accounting in a new folder; preserve original implementation. Remaining estimator ports stay queued. Bounded one-thread probes under 10 CPU-minutes. |
| T19: **Finish crossover reporting and find a low-compute mechanistic winner** | Codex lead + theory/literature/candidate sub-agents (session 216ca1055f0c81918ec742d3f3a2161f) | 2026-10-16 | [m16/approaches/crossover_compute/](m16/approaches/crossover_compute/) | light | Resumed at user request to pursue a solution autonomously. Parallel protocol/source review, analytic candidate design and independent falsification. Do not duplicate held T1 stochastic-margin work. Single-threaded bounded probes; no heavy-slot claim. |
| T4: **Re-score leaderboard v2 original-model rows** with the pooled 28M truth | DRI (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/leaderboard.py](m16/leaderboard.py) (shared code: DRI-only) | light | v2's orig rows used the old truth (~0.025 pp low). Done when the orig rows in [results/leaderboard.json](results/leaderboard.json) are refreshed. Plan: superseded by the v3 re-run (truths read at report time, so rescoring is automatic) |
| T17: **Harsher noise levels** (sigma 0.01-0.02 x RMS) in the leaderboard, reported per level; more truth samples for large budgets | DRI (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/leaderboard.py](m16/leaderboard.py) (shared code: DRI-only) | heavy | "Significantly alter accuracy" may mean more noise than our 0.0025/0.005 |
| T15: **Score the full MSE-vs-compute curve** (~1k to 1M evals) and report where each method crosses sampling | DRI (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [m16/leaderboard.py](m16/leaderboard.py) (shared code: DRI-only) | heavy | ARC asks to match sampling across compute; v2 tested only 16k/64k (tube won at 64k and lost at 256k) |
| T22a: **Web search, ARC side**: everything public on AlgZoo / mechanistic estimation from ARC and third parties (blog, LessWrong/AF, GitHub alg-zoo commits/branches/issues, talks, papers citing AlgZoo, other attempts) | DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [lit/web_arc/](lit/web_arc/) | light | Rule 7 relaxed by the owner; log post-Jan-2026 ARC sources in lit/web_log.md |
| T22b: **Web search, methods side**: deterministic computation of output probabilities / Gaussian measure of ReLU-network preimages faster than Monte Carlo (probabilistic verification, polytope Gaussian measure, rare-event, heuristic estimators, symbolic dynamics of PL maps) | DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z) | 2026-10-17 | [lit/web_methods/](lit/web_methods/) | light | Output: ranked list of transferable techniques with a concrete plan for M_16,10, added as backlog rows |
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
| T22 | **Ready** (light; DRI-only shared code) | **Use per-model sampling variance at actual cost in v3 group ratios** | Independent T19 review found that `mean(p)*(1-mean(p))/mean(cost)` differs from `mean(p_i*(1-p_i)/cost_i)`; with equal costs it adds between-model variance and can flatter ratios. Done when the DRI selects/documents the estimand, checks heterogeneous-model cases, and recomputes affected reports | [m16/leaderboard.py](m16/leaderboard.py) (`table`); [crossover_compute](m16/approaches/crossover_compute/README.md) |
| T3 | **Ready** (heavy) | **Symbols from rotating-neuron gate switches** instead of records (Poincare's untried variant) | Ceiling test like E27: done when the reduced system's exact bias is measured on all 18 models | [sci_poincare](m16/approaches/sci_poincare/README.md) |
| T6 | **Ready** (heavy; needs LLM agents) | **Qwen H2 re-run**: >=100 controls (top neurons projected out, or dense rotation), 2-3 explainers per item, n-gram dedupe for judges | Resolves the "inconclusive" verdict on the essay's claim | [RESULTS.md](RESULTS.md), [review](review/README.md), ledger H-Q1..Q4 |
| T10 | Blocked: T2 | **Minimal pair: H=16, T=4 (calculation wins 38-266x) vs T=5 (7.7x)**: find exactly which error patches appear at T=5 that do not exist at T=4, then fix the calculation for T=5. Absorbs T2's 'why does it become heavy-tailed' and T13's diagnostics | Smallest instance of the open problem (simplification ladder). Done when a deductive estimator beats sampling at H=16, T=5 on the leaderboard, or the new patches are characterised precisely | [sci_tao](m16/approaches/sci_tao/README.md) (`scan_HT.json`), ledger H-E11/H-E12; `zoo_2nd_argmax(16, T)` |
| T11 | Blocked: T10 | **Climb the ladder**: once T10 works, step T=5->6->...->10 at H=16 and check it survives H=8/32; each step must beat sampling on the leaderboard before moving on | Turns a small-model solution into the M_16,10 solution, or shows exactly where it breaks | [m16/leaderboard.py](m16/leaderboard.py) (generalise to (H,T)) |
| T12 | Blocked: T10 | **Training-stage dimension**: track when the hard error patches form across checkpoints (`n_seqs` = 2^20..2^30) for H=16, T=5 and T=10 | Shows which learned feature creates the heavy-tailed bands | `zoo_2nd_argmax(16, T, n_seqs=...)` |
| T9 | Blocked: ARC sharing their method; T14-T17 | **Score ARC's method** on our harness when it is shared | The comparison the project was set up for. Needs ARC's estimator in `est(model, budget, rng)` form, or their numbers | [m16/arc_eval.py](m16/arc_eval.py), [m16/leaderboard.py](m16/leaderboard.py), ledger H-E4/H-E5 |
| T5 | Blocked: user approval for more compute | **Brute-force exact reference** (needs much more compute): first test whether exact 10-D cone masses can be computed fast and accurately on a sample of regions | Gives an exact answer to compare with ARC; benchmark only. Go/no-go after the cone-mass test | INDEX open questions; [cone_enum](m16/approaches/cone_enum/README.md); ledger H-M7 |
| T13 | Closed (merged into T10) | *(merged into T10: its error-family / positive-only diagnostics are tools for the minimal pair)* | - | - |
| T23 | **Ready** (light) | **Converged, multi-family and multi-seed scalar-budget sweep** | T7 found strong budget sensitivity but seven fits reached the iteration cap. Done when predeclared families and at least 3 source-model seeds yield held-out curves with fitted and structural complexity reported | [constant_budget](algzoo/constant_budget/README.md), E28 |
| T24 | **Ready** (light; needs LLM agents) | **Fresh blinded M32 investigators and category judges** | Frozen E5 categories were missing and completed retrospectively. Done when an equal-effort blinded model-size ladder independently scores labels and reconstruction | [constant_budget](algzoo/constant_budget/README.md), E5 |
| T25 | **Ready** (light) | **Audit total description length of investigator formulas** | Searched position subsets and chosen forms carry information beyond fitted floats. Done when a written complexity rule accounts for structural choices and applies consistently to original and swept labels | [constant_budget](algzoo/constant_budget/README.md), H-A1 |
| T26 | **Ready** (light) | **Energy/readout-matched position patching for n3/n13 and fast modes** | T8 orthogonal enrichment is not causal specificity. Done when controlled position swaps and on-manifold patches show selective mediation or bound its absence against energy/readout-matched controls | [mechanism_checks](m16/approaches/mechanism_checks/README.md), E29 |
| T27 | **Ready** (light) | **Gate-history phase transport including input forcing** | Common-gate phase slopes differ from final-gate eigenvalue predictions by about 4–8 degrees. Done when exact transport on matched histories explains those deviations or quantifies the remaining discrepancy | [mechanism_checks](m16/approaches/mechanism_checks/README.md), H-M1 |
| T28 | **Ready** (light) | **DEV-only rank-respecting joint-tail diagnostic** | E30 Gaussian/Jacobian candidate failed; closure, gate changes and marginal-union error remain confounded. Done when a predeclared DEV-only diagnostic separates these sources without tuning on leaderboard test models | [candidate_notes](m16/approaches/crossover_compute/candidate_notes.md), E30 |

## Recently finished (details in the INDEX registry)

T7 (E5 scoring + E28 budget sweep) and T8 (E29 mechanism checks) completed by Codex sub-agents. T19 bounded reporter/candidate study is E30; remaining integration/search returned to Backlog. No Codex task claim remains held.

T1 (stochastic tie-margin reduction: **negative**, E28), T21 (stratified sampler: one pooled rejection stream, candidates charged, ~1.1 FPE per sample), T14 + T16 (leaderboard v3 classes and compute rule, with DRI rulings on the [estimator audit](review/estimator_audit/README.md): cost as implemented, advice flags, rejection charged), T18 (leaderboard for any (H,T): `--H --T`; truths for H=16, T=4 and T=5 in `results/leaderboard_truths_H16_T{4,5}.json`), E10-E27, LB2 (leaderboard v2), R1 (review): see [INDEX section 3](INDEX.md#3-experiment-registry) and the
[approaches index](m16/approaches/README.md).
