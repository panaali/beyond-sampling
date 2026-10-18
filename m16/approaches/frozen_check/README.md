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

## Method
`frozen.py` (own code; T274b's code was not read) and `summary.py` -> frozen.json, summary.json, `frozen.log`. For every LB3 model,
400k paired inputs (rng 282). For each design and m = 0..8, the bias of the frozen-gate approximation vs the net is computed with
perfect integration, i.e. on the same inputs. The triple design follows T274b's protocol: the input's ranks 4..10 are moved into the
fixed arrangement, the approximation is scored against the net on the moved input, and design_bias = net(moved) - net(original).
`agreement_vs_T274b_hash.json` compares each row with T274b's table.

## Results (unverified)
Per-group RMS bias (pp), paired, perfect integration:

| design | m | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|---|
| full | 0 | 2.02 | 3.81 | 1.39 | 1.11 | 2.21 | 3.35 |
| full | 1 | 2.72 | 4.15 | 2.27 | 1.53 | 1.16 | 1.65 |
| full | 2 | 3.06 | 4.05 | 2.70 | 2.02 | 1.13 | 0.86 |
| full | 3 | 2.86 | 3.75 | 2.57 | 2.03 | 1.26 | 0.57 |
| full | 4 | 2.57 | 3.40 | 2.31 | 1.90 | 1.27 | 0.63 |
| full | 6 | 1.92 | 2.55 | 1.73 | 1.48 | 1.05 | 0.57 |
| full | 8 | 1.27 | 1.79 | 1.18 | 1.09 | 0.86 | 0.57 |
| triple | 0 | 1.29 | 3.08 | 0.88 | 1.41 | 2.59 | 3.77 |
| triple | 1 | 2.17 | 3.40 | 1.89 | 1.45 | 1.46 | 1.97 |
| triple | 2 | 2.51 | 3.40 | 2.34 | 1.88 | 1.19 | 1.08 |
| triple | 3 | 2.09 | 3.14 | 2.02 | 1.75 | 1.17 | 0.76 |
| triple | 4 | 1.59 | 2.85 | 1.57 | 1.48 | 1.07 | 0.74 |
| triple | 6 | 1.20 | 2.65 | 1.17 | 1.11 | 0.88 | 0.48 |
| triple | 8 | 0.39 | 1.76 | 0.32 | 0.25 | 0.18 | 0.26 |
| V1_sign | 0 | 1.07 | 1.87 | 0.54 | 1.30 | 2.55 | 3.47 |
| V1_sign | 1 | 1.71 | 1.96 | 1.28 | 0.91 | 1.27 | 1.83 |
| V1_sign | 2 | 2.06 | 1.80 | 1.70 | 1.20 | 0.77 | 1.00 |
| V1_sign | 3 | 1.80 | 1.49 | 1.54 | 1.17 | 0.73 | 0.57 |
| V1_sign | 4 | 1.45 | 1.15 | 1.25 | 1.00 | 0.67 | 0.48 |
| V1_sign | 6 | 0.71 | 0.56 | 0.61 | 0.50 | 0.39 | 0.24 |
| V1_sign | 8 | 0.00 | 0.15 | 0.01 | 0.01 | 0.03 | 0.02 |
| V2_mix | 0 | 1.25 | 2.16 | 0.67 | 1.21 | 2.43 | 3.30 |
| V2_mix | 1 | 1.89 | 2.34 | 1.46 | 0.99 | 1.20 | 1.67 |
| V2_mix | 2 | 2.23 | 2.25 | 1.88 | 1.35 | 0.83 | 0.89 |
| V2_mix | 3 | 1.96 | 1.95 | 1.72 | 1.32 | 0.85 | 0.56 |
| V2_mix | 4 | 1.59 | 1.62 | 1.40 | 1.12 | 0.78 | 0.54 |
| V2_mix | 6 | 0.78 | 0.99 | 0.69 | 0.57 | 0.44 | 0.27 |
| V2_mix | 8 | 0.01 | 0.28 | 0.02 | 0.02 | 0.02 | 0.02 |
| min m<=4, full | | 2.02 | 3.40 | 1.39 | 1.11 | 1.13 | 0.57 |
| min m<=4, triple | | 1.29 | 2.85 | 0.88 | 1.41 | 1.07 | 0.74 |
| min m<=4, V1_sign | | 1.07 | 1.15 | 0.54 | 0.91 | 0.67 | 0.48 |
| min m<=4, V2_mix | | 1.25 | 1.62 | 0.67 | 0.99 | 0.78 | 0.54 |
| 3 x b_max(256k) | | 0.40 | 0.53 | 0.46 | 0.56 | 0.74 | 0.88 |
| b_max(256k) | | 0.13 | 0.18 | 0.15 | 0.19 | 0.25 | 0.29 |

The design bias of my triple arrangement is 0.36 / 0.80 / 0.85 / 1.83 / 3.24 / 4.08 pp (T274b's hash arrangement: 0.10-0.33). My
fixed arrangement is cruder than theirs.

## Verdict
- **(a) Pre-registered agreement on the full-order design: FAILED (3/6 groups within 25%).** My full-order m = 0 is ~2x T274b's on
  orig / seed / s0.0025 (2.02 / 3.81 / 1.39 vs 0.96 / 1.84 / 0.52).
- **Diagnosis (post hoc, labelled as such).** T274b's "7,920 classes = top-3 triple x K" is 720 x 11. That is consistent with K = the
  number of positive inputs, i.e. their canonical inputs condition on the sign count. My V1 (ordering x k, conditional means given k
  positives) reproduces **T274b's full hash table within 25% on all 6 groups at every reported m = 0, 1, 2, 3, 4, 6** (mostly
  within 10%), and m = 8 where resolvable. Their full-order row likewise matches V1 m = 0 (6/6 within 25%). So the discrepancy is a
  definition (sign-count conditioning), not an error in either implementation. Unconditioned canonical gates (my full / triple)
  are up to 2x worse.
- **(b) Conclusion reproduced: DEAD** for every design. The min over m <= 4 of the per-group RMS bias exceeds 3 x b_max(256k) on orig
  AND seed for full and triple (pre-registered), and also for V1 and V2. T274b's kill holds.
- **Variants (Directive 8 go line): both dead.** V1 (sign-conditioned) *is* T274b's design (min m <= 4: 1.07 / 1.15 / 0.54 / 0.91 /
  0.67 / 0.48 vs 3 x b_max 0.39 / 0.53 / 0.45 / 0.57 / 0.74 / 0.88; orig, seed, s0.005 exceed). V2 (4-witness quantile mixture within
  ordering x k) is uniformly *worse* than V1 (orig 1.25, seed 1.62). Mixing witness gate patterns does not reduce the freezing bias.
- **So T274b can be marked dead (verified, scoped):** freezing gates at canonical (or conditional-quantile) witnesses of ordering,
  or ordering x sign-count classes, for the first 10-m steps with exact last m <= 4 steps leaves a per-group RMS bias of >= 0.5-1.6 pp,
  at least 3x the 256k lenient limit on orig and seed. Only m >= 7-8 (exact for most steps) gets close, which is the region problem
  again.

## Alternatives not ruled out
- Witnesses chosen from the weights (e.g. gate patterns maximising mass per class) rather than model-free quantiles.
- Classes finer than ordering x sign count (e.g. ordering x signs x a coarse magnitude bucket of x_(2)-x_(3)); the bias falls ~2x
  from unconditioned to sign-conditioned, so finer conditioning might go further, at a class-count cost.
- Frozen gates for the *middle* steps with exact *first* and last steps (only last-m tested).

## Compute accounting
30 models x 400k inputs x (net + 4 designs x 9 m, 7 gate sets) ~ 25 CPU-min; tables 4e6 model-free draws. Single thread.
