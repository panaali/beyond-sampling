# frozen_check — T282: independent check of T274b's frozen-gate kill (and its best untried variants)

↑ [TASKS](../../../TASKS.md) · T274b: branch claude/t218-tie-tail-labels, m16/approaches/lenient_bb/README.md (read for the
protocol only; its code was not read)

**All claims unverified.** Diagnostic only (sampling is used to *measure* an approximation's bias, as in T274b).

## Pre-registration (written 2026-10-18, before any T282 number was computed)
**Approximation a(m)** (T274b's definition, own implementation). For an input x with ordering pi, take a canonical input
c built from x's class (below), run the trained net on c to get its gate patterns g_1..g_10, and run x through the net
with gates frozen at g_t for steps t = 1..10-m (h_t = g_t * (W_hh h_{t-1} + W_ih x_t)) and the true ReLU for the last m
steps. Bias(m) = mean 1[a(m) correct] - mean 1[net correct], paired on the same 400k N(0, I_10) inputs per model (rng 282, own
stream), for all 30 LB3 models (leaderboard v3 `models()`). Per-group RMS bias = sqrt(mean over the group's models of bias^2).
m = 0..8.

**Class designs.**
- **full-order** (T274b's ideal design): c = expected order statistics E[x_(r)] of 10 i.i.d. N(0,1) placed at x's own ordering.
- **triple** (my own rank_exact-like design, not T274b's exact hash): c has E[x_(1..3)] at x's top-3 positions, and
  E[x_(4..10)] at the remaining positions in increasing index order (a fixed arrangement).
- Variant **V1, sign-conditioned** (T274b's untried "gates conditioned on signs"): class = (ordering, k = number of positive
  inputs). c = E[x_(r) | exactly k positive] placed at the ordering. This is a model-free table (11 x 10, from 4e6 i.i.d. draws).
  Given the ordering, the signs of *all* inputs are fixed by k, so this conditions on every sign, not only the last inputs.
- Variant **V2, multi-witness mixture** (T274b's "multi-witness gate mixtures per rank class"): within (ordering, k), 4
  deterministic witnesses at the conditional 20/40/60/80% quantiles of each order statistic (model-free table). The
  approximation's correctness is the average over the 4 witnesses' frozen-gate predictions (a probability).

**Agreement criterion (independent check).**
(a) The full-order m = 0 per-group RMS bias is within 25% of T274b's (0.96 / 1.84 / 0.52 / 1.31 / 2.53 / 3.45 pp) on >= 5 of 6 groups.
(b) The same conclusion: min over m <= 4 of the per-group RMS bias > 3 x b_max(256k) on orig AND seed for full-order and triple
(T274b's dead line), with b_max(B) = sqrt(10 p(1-p)/B) and the per-group mean p from the leaderboard truths.
Both hold -> T274b's kill is reproduced and can be marked dead (verified, scoped to frozen-canonical-gate designs with exact
last m <= 4 steps). Otherwise the discrepancy is documented.

**Variant go line (Directive 8 amendment, as T274b):** a variant earns an estimator build iff some m <= 4 has per-group RMS
bias <= b_max(256k) on all six groups (perfect integration; this is necessary, not sufficient). It is **dead** if the min over
m <= 4 exceeds 3 x b_max on any group. Otherwise inconclusive.
