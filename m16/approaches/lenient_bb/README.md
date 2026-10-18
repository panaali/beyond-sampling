# lenient_bb (T274b): can a non-sampling estimator reach the lenient bar (<= 10x sampling MSE) on all six LB3 groups?

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) ·
[branch_bound](../branch_bound/README.md) · [rank_exact](../rank_exact/README.md)

## Question
Directive 8's lenient bar: MSE ratio <= 10 against plain sampling at equal FLOPs, per leaderboard-v3 group
(orig, seed, s0.0025, s0.005, s0.01, s0.02). The DRI asked:
1. what branch_bound does, and which parts are inductive under Hilton's definition;
2. pre-register a go line;
3. fix branch_bound's failing noisy groups;
4. a weight-derived variant with no fixed nodes;
5. fallback: rank_exact with exact last steps.

## Step 0-1 findings (before any new estimator; `rescore.py`)
LB3 raw rows (`lb3_raw_snapshot.jsonl`) and truths (`lb3_truths.json`), copied from origin/ccr-6a61f25c-wn80ii
`results/`, were re-scored at **every** budget with cluster-bootstrap CIs. This is the same rule as
`m16/leaderboard.py table()`: per-run sampling MSE p(1-p)/charged cost, bootstrap over models.

**branch_bound** (`python rescore.py branch_bound`):

| budget | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|
| 1k = 4k = 16k = 64k (identical) | 5.26 | 8.70 [5.6, 11.6] | 5.23 [3.7, 7.0] | 12.8 [1.9, 26.0] | 18.9 [1.7, 43.1] | 9.51 [2.4, 16.9] |
| 256k | 4,668 | 8,067 | 4,714 | 10,268 | 15,373 | 8,326 |

- **What it is at <= 64k** (`branch_bound/estimator.py`): the 180 root cones (a = argmax position, b = 2nd,
  s = sign x_b), with exact model-free Gaussian masses. There is **one model evaluation at each cone's centre**,
  and a cone counts as correct iff its centre is classified correctly.
  - Charged cost is 180 FPE at every budget below ~144k, so the estimate and ratio do not change with budget.
  - Errors are +3.6 pp (orig), +3 to +9 pp (seeds), and -3.5 to +4.8 pp (s0.0025).
  - From 256k the real B&B runs, the centre rule stays biased (E-branch_bound: -16 pp at 1e11 FLOPs), and the
    ratio explodes.
- **Why "quadrature":** it is a fixed-node cubature, 180 fixed witness points weighted by exact cone masses.
- **Inductive vs deductive (Hilton, `lit/web_arc/hilton_talk_2026-09_notes.md`: "not allowed to just sample some
  inputs ... and assume that other things will behave similarly"; constructed inputs are allowed if one
  "extrapolates deductively"):**
  - cone masses: **deductive** (model-free, exact);
  - centre evaluations: constructed inputs, allowed;
  - assigning the whole cone the centre's verdict: **inductive**. It extrapolates observed behaviour at one input
    to the region with no argument from the weights.

  So branch_bound at <= 64k is inductive in the decisive step.
- **sci_tao already passes the lenient bar** (`python rescore.py sci_tao`): ratio 0.01-2.9 at every budget and
  group, upper CI <= 5.5. It is the same class (fixed Sobol nodes, inductive by the same test). So
  "branch_bound is the only non-sampling row near 10x" holds only if inductive quadrature rows are excluded, and
  then branch_bound is excluded too.
- **The v3 lenient bar is degenerate for cheap estimators.** v3 scores each run at its own charged cost C, and
  ratio = bias^2 C / p(1-p).
  - Any estimate within sqrt(10 p(1-p)/C) passes: +-5 pp at C = 180, +-21 pp at C = 1.
  - A constant 0.95 charged at 1 FPE passes all six groups.
  - "Fixing" s0.005/s0.01 for branch_bound is trivial by evaluating fewer cones (lower C). That would be gaming,
    and is not done here.

## Pre-registration (written 2026-10-18 before building or measuring any new estimator)
**Go line (the bar used here).** A frozen method passes iff, at one budget B in {16k, 64k, 256k} stated in
advance:
- its charged cost (T196 FLOPs, multiply-add = 2, as implemented) is >= B/2;
- the upper 95% cluster-bootstrap CI of the MSE ratio is <= 10 on all six groups (the strict bar, <= 1, is also
  reported);
- every step is labelled deductive or inductive (Hilton).

The required |bias| per group is b_max = sqrt(10 p(1-p)/B), with the per-group p. At p(1-p) = .045 that is
**0.53 / 0.27 / 0.13 pp** at 16k / 64k / 256k (lenient). The strict bar is sqrt(p(1-p)/B): 0.17 / 0.08 / 0.04 pp.
*Correction 2026-10-18:* the first version of this README quoted the strict figures as lenient. The DRI caught it
before any result was scored.
**DRI approval:** this go line is now the board ruling (Directive 8 amendment, ccr 4df09663).

**Feasibility test for the fallback (rank_exact + exact last m steps), before building it.**
- rank_exact freezes all gates at the class's canonical input (7,920 classes = top-3 position triple x K, other
  ranks in a fixed hashed arrangement). Its cost is already ~87k FPE (7,920 x 11 tangent passes), so it can only
  be scored at B = 256k: **b_max ≈ 0.13 pp** at p = .953 (lenient; larger for low-accuracy groups).
- The approximation's bias (with perfect integration) is measured by sampling, as a *diagnostic only*:
  - a(m) = gates frozen at the canonical input for steps 1..10-m, true ReLU for the last m steps, m = 0..4;
  - bias(m) = E[correct_approx] - E[correct_net], paired on 400k inputs per model, all 30 LB3 models;
  - two designs: "triple-hash" (rank_exact's classes) and "full-order" (canonical input of the input's own
    complete ordering: an ideal lower bound on gate-freezing bias, infeasible as an estimator at 10! x 11
    classes).
- **Go** for building it: some m with per-group RMS bias <= b_max(256k, per-group p) on all six groups,
  triple-hash design.
- *Added at the DRI's request, same rule:* a **"cone" design**, the cheapest fully deductive baseline. It keeps
  branch_bound's 180 root cones (a = argmax, b = 2nd, s = sign x_b) and exact masses. The centre-evaluation verdict
  is replaced by gates frozen at a weight-free canonical input of the cone (model-free conditional means of
  max / 2nd / rest given s, placed at a / b / others), with exact model-free integration of the frozen-linear
  network over the cone. Cost ~180 x 11 FPE plus quadrature, so it is judged at B = 16k with b_max = 0.53 pp.
- **Dead:** min over m of the per-group RMS bias > 3 x b_max on any group.
- **Alternatives:**
  - (i) bias could be small on average but cancel across models; the per-group RMS guards against this;
  - (ii) exact last steps might fix the 2nd-vs-3rd comparison but not the blind region; report bias by case.

## Results (2026-10-18)

### Lenient-bar bias limits b_max = sqrt(10 p(1-p)/B), per-group mean p(1-p) from `lb3_truths.json`

| B | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|
| 16k | 0.53 | 0.71 | 0.61 | 0.75 | 0.98 | 1.18 |
| 64k | 0.26 | 0.35 | 0.31 | 0.38 | 0.49 | 0.59 |
| 256k | 0.13 | 0.18 | 0.15 | 0.19 | 0.25 | 0.29 |

### Fallback: rank_exact + exact last m steps — **DEAD** (`python frozen_bias.py 400000`, `frozen_bias.log`, `results/frozen_bias.json`)
Per-group RMS bias (pp) of the frozen-gate approximation with perfect integration, paired on 400k inputs per model,
all 30 LB3 models. "hash" = rank_exact's classes on inputs moved into its arrangement; design_bias = that
arrangement's own error.

| | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|
| design_bias (hashed arrangement of ranks 4-10) | 0.10 | 0.21 | 0.15 | 0.22 | 0.33 | 0.15 |
| m = 0 (= rank_exact) | 0.89 | 1.79 | 0.48 | 1.39 | 2.67 | 3.49 |
| m = 1 | 1.48 | 1.86 | 1.14 | 0.91 | 1.40 | 1.87 |
| m = 2 | 1.91 | 1.73 | 1.61 | 1.15 | 0.81 | 1.01 |
| m = 3 | 1.70 | 1.42 | 1.51 | 1.14 | 0.72 | 0.55 |
| m = 4 | 1.35 | 1.11 | 1.18 | 0.94 | 0.63 | 0.44 |
| m = 6 (beyond pre-registration) | 0.65 | 0.57 | 0.57 | 0.46 | 0.38 | 0.22 |
| m = 8 (beyond pre-registration) | 0.001 | 0.17 | 0.01 | 0.02 | 0.03 | 0.03 |
| full-order design, m = 0 (ideal per-ordering classes) | 0.96 | 1.84 | 0.52 | 1.31 | 2.53 | 3.45 |

- **Pre-registered verdict: DEAD.** The minimum over m = 0..4 exceeds 3 x b_max(256k) on every group: orig 0.89
  vs 0.40; seed 1.11 vs 0.53; s0.01 0.63 vs 0.74 is the closest, but orig and seed already fail.
- **Exact last steps do not help at small m.** On orig and seed, m = 1-3 is *worse* than m = 0. The frozen early
  state is inconsistent with exact late gates, and the two errors stop cancelling.
- Only m = 8 (just the first two steps frozen) gets the freezing bias under b_max. But "exact last 8 steps" is the
  original 8-step region problem (T222: ~15x more regions per step), so it is not a route.
- Per-ordering classes do not help either: the full-order design is no better than the 7,920 hashed classes, so
  the bias is gate-freezing, not the class design. The design bias alone (0.33 pp on s0.01) already exceeds that
  group's 256k limit.

### DRI-suggested deductive cone baseline — **fails at 16k** (`python cone_bias.py 400000`, `results/cone_bias.json`)
branch_bound's 180 root cones with exact masses, verdict from weights (gates frozen at a model-free canonical input
of the cone), perfect integration:

| | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|
| RMS bias, m = 0 (pp) | 1.64 | 17.1 | 2.87 | 5.26 | 9.18 | 11.6 |
| best over m = 0..8 | 1.64 | 0.99 (m=8) | 2.12 | 1.63 | 2.40 | 2.31 |
| b_max(16k) | 0.53 | 0.71 | 0.61 | 0.75 | 0.98 | 1.18 |

A cone is far too coarse a class for one gate pattern. On the seeds the frozen gates of a single canonical input
miss by 17 pp.

### Branch_bound itself under the approved go line
- At B >= 16k the method must spend >= 8k FPE, so the 180-centre rule is out (cost 180).
- Its B&B centre rule at higher compute stays biased by +2 to -16 pp (branch_bound README), against
  b_max = 0.53-1.18 pp. It cannot pass any group at 16k-256k.
- "Fixing" s0.005/s0.01 at the old v3 scoring was possible only via the cost-floor loophole.

## Status
Done (2026-10-18). **No frozen method passes the approved go line**, so nothing was submitted for HP1. The closest
attempts are:
- **rank_exact (m = 0):** bias 0.48-3.5 pp, 4-14x over b_max(256k).
- **rank_exact + 6 exact last steps:** 0.22-0.65 pp, 1.5-5x over, and not cheap.

The inductive quadrature row sci_tao already passes the lenient bar under the old scoring, and needs a re-check
under the cost floor (its charged cost is about the budget, so probably fine).

## Compute accounting
- Re-scoring: seconds.
- Diagnostics: 30 models x 400k inputs x (2 designs x 7 m + 1) and 30 x 400k x 6 (cone), about 3e9 sequence
  forwards in all. These are *diagnostic samples, not estimator cost*.
- About 70 CPU-min, one heavy process at a time, < 1 GB.

## Claims (unverified)
- [measured] branch_bound's LB3 ratios at 1k-64k come from one fixed 180-FPE centre rule (biases +3 to +9 pp).
  Under Hilton's test it is inductive in the decisive step; so is sci_tao.
- [measured] Under v3's charged-cost scoring the lenient bar is met by any estimate within sqrt(10 p(1-p)/C), a
  loophole now closed by the DRI's B/2 cost floor.
- [measured] Frozen-gate (rank_exact) bias with perfect integration is 0.48-3.5 pp per group. Running the last
  1-4 steps exactly does not reduce it below 0.44 pp anywhere, and it increases it on orig and seed.
- [measured] The fully deductive cone baseline is biased 1.6-17 pp.
- [inferred] No frozen-gate/witness deduction at the granularity of orderings or cones reaches the lenient bar at
  16k-256k. The error lives in gate flips within classes, consistent with T272 (freezing in-band gates: 2-21 pp)
  and T222/T218 (no small set of regions or families carries the mass).

## Alternatives not ruled out
- Freezing at several witnesses per class (a mixture over gate patterns), or conditioning gates on the last few
  inputs' signs. Not tested; the m = 6 numbers suggest at least 4 early steps need exact gates.
- A non-frozen deductive approximation (moment propagation: ADF failed in rank_exact; ARC cumulants 711-7,887x)
  remains far off.
- The abs-sigma .05 HP1 regime (accuracy .12-.28) was not examined, because nothing passed on LB3.
