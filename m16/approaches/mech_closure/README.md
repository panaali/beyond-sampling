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

## Status
Pre-registered; not run.
