# T338: a mechanism-informed closure of the low inputs: which lows does each model's W say matter?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T241 neuron account](../neuron_account/README.md) · [T298](../r0_variants/README.md)

## Question
T317e treats the low inputs with closure c: v4 exact at the latest free position, and the other 6 lows pooled at
μ(v4). This closure carries 34% of the MSE, and its floor (RMS .463 pp, T324 E0; .471 in T304) blocks strict.

T241 found time-local neuron roles: recency, a running max, a leave-one-out max; n2 peaks at t = 4–5, and n5 matters
from t = 2. Each retrain implements these roles with different neurons. If the roles make only a few low positions
matter for the late margins, then giving those lows exact treatment, chosen per model from W, should lower the floor
at small cost.

The task asks for the oracle floor of such a closure, on 80 DEV2 models with T298's protocol, and the cost in B2
prefix-tree nodes per line.

## Pre-registration (2026-10-18, before any run)
**Closure family M(A).** A is an allocation: a set of (triple, region, position j) with j a free position of the
triple. For an input x with skeleton (p1, p2, p3, v1, v2, v3) and region r (nb if v2 > 0, else b), let S be the
positions allocated to x's triple and region:
- positions in S keep their own value (oracle: x's true value);
- the remaining free positions R = F \ S get closure c on R: the max of x over R is placed at the latest position of R
  with its true value, and the others are set to μ(that max);
- with A = ∅ this is exactly closure c (E0 in T324's terms).

T327's fixed spread pattern for the lows is a different design and is not tested here.

**The allocation is derived from W alone**, with no sampling of model inputs and no truths.
- The anchor for each triple and region is the closure-c point at the region's model-free transverse values: v1*, v2*,
  v3*, v4* at the latest free position, lows at μ(v4*). The constants are T317e's TABLES, from `cont_e.py`, a verbatim
  copy of @01906d33.
- Margin = L_p2 − max_{k ≠ p2} L_k, from one forward pass of the model.
- Score of (triple, region, j): Δ = max over q ∈ {Q(.1), Q(.9)} of the law N(0,1)|<v3* of |margin(anchor with x_j = q) −
  margin(anchor)|. These are exact forward passes, so gate switches are included, not linearised (T319).
- **Node cost.** Each allocated low is integrated in the estimator by a K = 2 Gauss rule of its law, which doubles
  that triple's rows from step j onward. Estimated added nodes: (T − j) per allocated low.
- **Budget greedy.** Per region, add pairs in decreasing Δ/(T − j), at most one position per triple and region, while
  the estimated nodes stay within budget. The budget is +30% of 2,815 = 845 nodes per line (the DRI's go line).
- Then the exact executed count: I build each line's inputs with the K = 2 expansion and count unique prefix nodes
  with T317e's own `first_cell` rule. If any line exceeds 3,660 nodes, I drop the lowest-ranked pairs until all lines
  fit.

**Variants, all on the same x (paired):**
- (0) closure c, the baseline (A = ∅). It should reproduce E0 (RMS .463, bias −.154) within noise.
- (1) **S_late**: the latest free position of each triple exact, a model-free recency rule, under the same budget
  rule and order of lateness.
- (2) **S_mech**: the W-derived budget greedy above. **This is the primary variant.**
- (3) **S_adj**: lows adjacent in time to p2 or p3 (p ± 1, free), in order of lateness, under the same budget.
- (4) **S_mech, no budget**: the best Δ position for every triple and region. This is a potential, not a design.

Secondary, K = 2 realisability: (1) and (2) with the allocated lows at the two Gauss nodes of N|<v3 (rule tabulated
on a v3 grid), with ok weighted by the node weights instead of the true value.

**Protocol (T298).**
- 80 DEV2 models; x ~ N(0, I_10), N = 500k per model, seed 338000 + index.
- b_m = mean(ok_closure − ok_true).
- Reported: RMS_m b_m (raw and noise-corrected), mean b, slope on truth, nodes per line (max over lines and the mean),
  and allocated pairs.
- Paired bootstrap over models (10,000) for RMS(2) − RMS(1) and RMS(2) − RMS(0).

**Go (DRI):** (2) has raw RMS floor ≤ .40 pp **and** ≤ 3,660 nodes on every line.
- "Mechanism-informed" is claimed only if (2) also beats (1) by more than the 95% bootstrap CI (the upper CI of
  RMS(2) − RMS(1) < 0).
- Otherwise the gain, if any, is recency, not mechanism.

**Alternatives first.**
1. Recency alone, through the last input, carries everything, and W adds nothing. Variant (1) tests that.
2. The closure floor sits in lows that switch gates jointly across many positions (T319: only 4–12% of draws stay in
   the anchor's cone). Then one exact low per triple helps little whatever the selection, and (4) shows the ceiling.
3. The anchor's Δ may rank positions poorly away from the anchor. Then (4) beats (2) only weakly, and (2) ≈ (1).

**Amendment (DRI, 2026-10-18, from T327).** This was made while the 80-model run was in progress. I had seen the
first model's numbers only, no aggregate.
- T327 (claude/t286-cheap-frontier ca5c94d3) found that a fixed spread of the lows (μ + (−1)^j σ, hard verdict) lowers
  the oracle floor of the deployable closure from .482 to .343 pp, and the reverse pattern to .255.
- **The amended go line: S_mech counts only if its oracle floor is < .255 pp at ≤ 1.3x nodes.** The original
  ≤ .40 line is reported as well.
- Caveat on comparability: T327's .482 baseline is its own deployable closure. Mine is closure c with v4's true value
  (T324 E0, .463). Floors are compared as reported, with this caveat.
- I will also say whether the spread's gain has a mechanistic reading in the T241 account.

## Status
**Done. NO GO** on both the original (≤ .40) and the amended (< .255) line. W-derived selection does not beat
recency. Unverified; the floors are diagnostic sampling.

## Results
To reproduce: `PYTHONPATH=<shim> python3 mc.py` (80 DEV2 models × 500k inputs, ~25 CPU-min), then `python3 analyze.py`.
Outputs are `results/mc_all.json` and `results/summary.json`. The node counts use T317e's prefix rule on the K = 2
expanded lines; the base is 2,815 and the budget 3,660.

| variant | oracle floor RMS (pp) | mean (pp) | slope | max nodes per line | lows made exact per model |
|---|---|---|---|---|---|
| (0) closure c (baseline; T324 E0 .463, T304 .471, T327 R1W .481) | .473 | −.153 | −.013 | 2,815 | 0 |
| (1) S_late: latest free low exact | .426 | −.088 | −.012 | 3,658 | 1,306 |
| **(2) S_mech: W-derived, budget greedy** | **.410** | −.065 | −.014 | 3,659 | 630 |
| (3) S_adj: lows next to p2/p3 | .423 | −.069 | −.005 | 3,631 | 812 |
| (4) S_mech, no budget (best position per triple) | .309 | −.132 | −.008 | **6,813 (2.4x)** | 1,440 |
| (1) at K = 2 nodes (realisable) | .422 | −.055 | −.014 | | |
| (2) at K = 2 nodes (realisable) | .422 | +.014 | −.023 | | |

The lows count is per model, over 720 triples × 2 regions.

Paired bootstrap over models (10,000) for differences in RMS:
- (2) − (1) = −.017 [−.051, +.017];
- (2) − (0) = −.064 [−.099, −.025];
- (2) − (3) = −.013 [−.048, +.022].

- **No go.**
  - On budget, the W-derived closure lowers the floor by .064 pp (.473 → .410). That misses both .40 and the amended
    .255 (T327's zero-cost reverse spread pattern).
  - It is not distinguishable from the model-free recency rule (1) or the adjacency rule (3), so the gain is
    recency, not mechanism.
  - At K = 2 nodes, the realisable version, (1) and (2) are equal (.422).
- **Even without a budget the ceiling is .309.** One exact low per triple, at the position W ranks highest, at 2.4x the
  nodes, is still worse than T327's zero-cost spread (.255). Alternative 2 holds: the floor is not in any single low
  position.
- **Where W says the lows matter.** Positions picked by the no-budget rule, summed over 80 models × 1,440:

  | position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | count | 24.1k | 18.9k | 14.9k | 11.9k | 9.4k | 7.4k | 6.1k | 5.8k | 7.2k | 9.6k |

  The most-picked are the *earliest* positions, then a small recency bump at 8–9. The node budget forces the greedy
  onto late positions (27k of 50k picks at position 9), because an early exact low duplicates almost the whole
  prefix tree. The positions that matter most at the anchor are therefore the most expensive to treat exactly in
  this tree.

**A mechanistic reading of T327's spread gain** [speculative, consistent with the above]:
- In the T241 account the late margins read the lows through order-statistic neurons: running max, leave-one-out max
  and recency. They do not read any one low.
- Closure c sets 6 lows to one value μ(v4). The max of the pooled lows then equals μ(v4), far below a typical
  second-largest low (≈ μ + σ). The running-max and leave-one-out-max neurons see an unnaturally flat background.
- A ± σ spread restores a realistic max and spread of the lows at every point in time, which those neurons respond
  to. That would explain why a position-pattern change with no per-model selection beats any one-position exact
  treatment. It would also explain why the earliest positions rank highest here: the running max is set early and
  carried forward.
- **Cheapest test (not run):** an oracle closure c where the lows keep their true *rank profile* (true second-max at
  its true position, others at μ). Against T327's spread, this separates "max of the lows" from "alternation per se".

## Compute accounting
`mc.py`: per model, the allocation (720 × 7 × 2 × 2 anchor forward passes, plus node counts), then 500k inputs × 10
closure evaluations. About 25 CPU-min in total, single thread, `ulimit -v 2500000`. Diagnostic sampling only;
nothing here is an estimator.

## Claims
- [measured] W-derived exact lows under a 1.3x node budget lower closure c's oracle floor from .473 to .410 pp. This is
  not distinguishable from latest-low (.426) or adjacency (.423) rules.
- [measured] Without a budget, one exact low per triple reaches .309 pp at 2.4x the nodes. This is still above
  T327's zero-cost spread (.255).
- [inferred] W ranks early low positions highest for the anchor margin. The exact-low design cannot afford those
  positions in a prefix tree.
- [speculative] The spread's gain comes from restoring the lows' order statistics, which the max-type neurons read.

## Alternatives not ruled out
- Δ at one anchor may misrank positions away from the anchor (alternative 3). (4) uses the same Δ, so a better
  ranking could lower (4) below .309.
- More than one exact low per triple was not tested (the budget forbids it).
- Combining with T327's spread (spread lows plus W-chosen exact lows) was not tested. That belongs with T339's design.
