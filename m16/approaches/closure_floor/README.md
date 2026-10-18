# T327: closure floor, error side. Re-check of T319's smoothed-closure negative and its two untried variants

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) ·
[T286 cheap frontier](../cheap_frontier/README.md)

## Question
The strict bar at T317e's ~5.4k FPE needs RMS <= .37 pp. The closure oracle floors (R0 hard .53, top-4 .40, R1W .46 pp;
INDEX T298/T304/T307/T319) are above that, so the closure binds. T319 found that a Gaussian-smoothed verdict
1[m > 0] -> Phi(m / (s ||J||)) makes the floor worse. Its stated cause: J is taken on one gate pattern, while the lows
switch gates. (1) Does T319's failure reproduce with independent code? (2) Do its two untried variants lower the floor:
(a) J clipped to the gate cone, (b) an exact 1-D treatment of the largest-|J| low with the rest kept hard?

## Pre-registration (2026-10-18, before any code was written)
Read before writing this: INDEX rows T292, T296-T309, T317, T319 (setup only). **`smooth_verdict/` was not read** and
will not be read until the re-check (step 1) is pushed.

**Closures** (class skeleton = positions p1, p2, p3 and values v1 > v2 > v3 of the top 3; "lows" = the 7 other inputs,
i.i.d. N(0,1) | < v3 given v3):
- R0: all 7 lows at m0(v3) = E[N(0,1) | < v3].
- top-4: v4 at its true position and value; the 6 lowest at E[N | < v4].
- R1W (T304 stage A): v4's true value placed at the latest position not in the triple; the 6 lowest at E[N | < v4]
  elsewhere. This is the closure T317e uses, so it is the one that counts for the go line.

**Verdicts at the closure point xbar** (target = position of rank 2; L = logits; free lows = the closure's
mean-valued inputs, sd s = sd(N | < U), U = v3 for R0 and v4 for top-4 / R1W):
- hard: 1[argmax L(xbar) = target].
- smooth (T319's form, as described in its INDEX row): m = L_target - L_r*, with r* the strongest rival at xbar,
  J = dm/dx over the free lows on xbar's gate pattern, P = Phi(m / (s ||J||)). Also the product over all 9 rivals,
  prod_k Phi(m_k / (s ||J_k||)).
- (a) cone-clipped: each free low t enters as J_t clip(x_t, lo_t, hi_t), where [lo_t, hi_t] is the exact interval of
  x_t (others fixed) on which xbar's gates do not change. Margin mean m + sum_t J_t (E clip - xbar_t), variance
  sum_t J_t^2 Var clip under N | < U (64-node quantile rule, model-free), P = Phi(mean / sd). Max-rival form and
  product form.
- (b) exact 1-D: t* = argmax_t |J_t| over the free lows. The verdict along x_t* (others at xbar) is resolved exactly
  by walking its piecewise-linear segments from U down to -8 (mass below is < 1e-15). P = mass of the correct set
  under N | < U. The other lows stay hard at xbar.

**Oracle protocol (paired; diagnostic, uses sampled inputs).** Per DEV2 model (all 80), n = 200k i.i.d. N(0,1) inputs.
For each input, the closure point is built from its own top-3 (and v4), and every verdict is computed. Closure error
per model = mean verdict - mean network correctness on the same inputs (paired; MC s.e. ~.02-.04 pp), plus
(network sample mean - official T284 truth), i.e. closure mean - truth. Report RMS over the 80 models, mean, and the
split blind (v2 <= 0) / non-blind.

**Re-check criterion (step 1).** T319's failure counts as reproduced if my hard floors are within 10% of T319's
(R0 .527, top-4 .399) and the max-rival smooth / hard RMS ratio is > .70 (T319's go line) for both closures.

**Cost model for the go line** (FPE = 5,948 FLOPs, multiply-add = 2). Extra work per closure point beyond the hard
verdict: smooth = 1 reverse pass for J (~1.5 FPE per rival); (a) + a forward-mode tangent over the free lows and
interval work (~1 + 6-7 columns, ~8.5 FPE); (b) the number of 1-D segments walked x ~1 FPE (counted per input by the
walk). It is charged at two deployment granularities: once per triple (720 points per model) and once per walk cell
of T317e's estimator (the number of cells is taken from r1_cont when costed; reported, not tuned).

**Go line (DRI):** a variant with R1W oracle RMS <= .30 pp on all 80 DEV2 AND <= 1.5k FPE added at the per-triple
granularity. **Stop:** no variant reaches <= .40 pp on R1W (no useful gain over the .46 floor).
**Expected (before running):** smooth reproduces T319 (worse than hard). (a) removes most of the soft-tail damage but
does not beat hard by much, because the 2~3 band, not the tail, is what a first-order law can fix. (b) gains <= 10%,
because the largest-|J| low carries only part of the low variation. My guess: no go.

**Alternatives kept in view.** (1) A variant could improve RMS by cancelling bias across regions; report the region
split. (2) Paired MC noise (~.03 pp) is small against the .30 line. (3) The oracle evaluates each variant at the true
skeleton of every input; an estimator evaluates it at quadrature nodes, so a passing oracle is necessary, not sufficient.

## Method
(after the runs)

## Status
Pre-registered; running.
