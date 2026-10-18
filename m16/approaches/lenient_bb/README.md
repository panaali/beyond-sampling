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

The required |bias| per group is about b_max = sqrt(10 p(1-p)/B). At p = 0.95 that is 0.17 / 0.084 / 0.042 pp
at 16k / 64k / 256k (larger for the low-accuracy s0.02 copies).

**Feasibility test for the fallback (rank_exact + exact last m steps), before building it.**
- rank_exact freezes all gates at the class's canonical input (7,920 classes = top-3 position triple x K, other
  ranks in a fixed hashed arrangement). Its cost is already ~87k FPE (7,920 x 11 tangent passes), so it can only
  be scored at B = 256k: **b_max ≈ 0.04-0.10 pp**.
- The approximation's bias (with perfect integration) is measured by sampling, as a *diagnostic only*:
  - a(m) = gates frozen at the canonical input for steps 1..10-m, true ReLU for the last m steps, m = 0..4;
  - bias(m) = E[correct_approx] - E[correct_net], paired on 400k inputs per model, all 30 LB3 models;
  - two designs: "triple-hash" (rank_exact's classes) and "full-order" (canonical input of the input's own
    complete ordering: an ideal lower bound on gate-freezing bias, infeasible as an estimator at 10! x 11
    classes).
- **Go** for building it: some m with per-group RMS bias <= b_max(256k) on all six groups, triple-hash design.
- **Dead:** min over m of the per-group RMS bias > 3 x b_max on any group.
- **Alternatives:**
  - (i) bias could be small on average but cancel across models; the per-group RMS guards against this;
  - (ii) exact last steps might fix the 2nd-vs-3rd comparison but not the blind region; report bias by case.
