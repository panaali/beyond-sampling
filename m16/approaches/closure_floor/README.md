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

## Addendum pre-registration (DRI / Fable r5, 2026-10-18 12:13, while the main oracle was running; before any spread run)
The main pre-registration above is unchanged.

**(A) Decision in cost terms.** Report each variant as projected DEV2 MSE x C relative to T317e. T317e (= T317c estimates):
DEV2 RMS .795 pp at C 5,435 FPE, so MSE x C = 3,435 pp^2 FPE. Projection, declared as such: a closure change from floor
f0 to f1 moves T317e's MSE by (f1^2 - f0^2), with the rest (integration and the other declared approximations) unchanged,
and C becomes 5,435 + added FPE. Decision: worthwhile only if the projected MSE x C <= .85 x 3,435 = 2,920.

**(B) One-pattern spread lows (separate arm).** Hard verdict at a closure point where each free low at absolute position
j in 0..9 is set to mu + (-1)^j sigma, with mu, sigma the mean and sd of N(0,1) | < U (one fixed, position-deterministic
pattern; no extra cost because the lows still take one value per position pattern). Run for R1W (decides), plus R0 and
top-4 (information). Same 80 DEV2 models, own seeds, 100k inputs per model, paired with the network, against the official
truths. **Go: R1W spread floor <= .40 pp.** The reverse pattern mu - (-1)^j sigma is reported too, as a check that the
gain is not a sign accident (it does not count for the go line).

**Deviation, declared:** the main oracle runs at n = 100k inputs per model, not 200k (CPU budget). Paired MC s.e. of a
closure error ~.05 pp, still small against the .30 / .40 lines.

## Method
`oracle.py` (independent code, written without reading smooth_verdict/): per input, the closure point from the input's
own skeleton; one forward pass with a 10-column forward-mode tangent gives the logit Jacobian A and, per input
coordinate, the exact interval on which the closure point's gates do not change (`fwd_full`). Verdicts as
pre-registered. (b)'s walk (`walk_1d`) re-evaluates the network at each cell along x_t* (one forward + a 1-column
tangent), finds the next gate flip below, and adds the exact truncated-normal mass of the interval where all 9 rival
margins are positive. Checked against 400-draw MC along the same lines on 2,000 points: mean difference 1e-4, max
.057 (MC s.e. ~.025). `spread.py` is arm B. `score.py` adds (network sample - official truth) to the paired
differences (`results/score.txt`); `results/cost_decision.txt` holds the addendum projection.

## Results (80 DEV2, 100k inputs per model; network sample vs official truth RMS .105 pp)
Reproduce: `python oracle.py 100000; python score.py; python spread.py 100000` (about 60 + 3 CPU-min).

### Step 1: re-check of T319 (pre-registered criterion: hard floors within 10% of T319, smooth/hard ratio > .70)
| closure | hard RMS (T319) | smooth, max rival (T319) | ratio | product over rivals (T319) |
|---|---|---|---|---|
| R0 | .532 (.527), +1% | 1.329 (1.326) | 2.50 | 11.69 (11.7) |
| top-4 | .416 (.399), +4% | .662 (.657) | 1.59 | 5.87 (5.9) |

**T319's failure reproduces** with independent code and draws: both criteria are met, and the smooth and product
numbers agree with T319's to within 1-5%. The smooth verdict turns R0's mean from +.23 to -.98 pp (too many
errors predicted in the non-blind region), as T319 describes.

### Step 2: the two untried variants
| closure:verdict | RMS pp | mean pp | blind mean | non-blind mean / RMS | +FPE per triple x 720 |
|---|---|---|---|---|---|
| R0 hard / (a) clip / (b) walk1d | .532 / .387 / .354 | +.23 / -.11 / +.10 | +.04 / +.05 / +.03 | +.18/.51, -.16/.41, +.06/.34 | 0 / 6.1k / 18.9k |
| top-4 hard / (a) / (b) | .416 / **.195** / **.226** | +.31 / +.04 / +.13 | +.03 / +.03 / +.02 | +.27/.37, -.01/.17, +.10/.19 | 0 / 6.1k / 14.7k |
| **R1W hard / (a) / (b)** | **.481 / .553 / .440** | -.14 / -.37 / -.23 | +.03 / +.04 / +.03 | -.18/.50, -.41/.59, -.27/.46 | 0 / 6.1k / 18.0k |
| (a) product form, R0 / top-4 / R1W | 6.07 / 3.10 / 3.03 | -4.5 / -2.3 / -2.0 | | | 15.8k |

(b) walks 9-12 cells per point on average (x 2 FPE as implemented).
- **Go line: NOT met.** On R1W, the closure T317e uses, neither variant reaches .30. Both also miss the .40 stop line:
  (a) is worse than hard (.553), and (b) gains 9% (.440). Both cost 6-18k FPE per model even at one point per triple,
  against the 1.5k allowed. **Stop for (a) and (b) on R1W.**
- Cone-clipping fixes the smooth verdict's failure where the closure keeps v4's true position: top-4 .416 -> .195 and
  R0 .532 -> .387. It hurts R1W. In R1W, v4's value sits at a position that is not its own, so the first-order law
  around that point is biased more (mean -.37 pp). This supports T319's cause (one gate pattern for switching lows):
  clipping to the cone removes most of the soft-tail damage. The product form stays far too pessimistic in all three
  closures (the rival margins are strongly dependent).

### Arm B (addendum): one-pattern spread lows, hard verdict (`results/spread.txt`)
| closure | mean lows | spread + (pre-registered) | spread - (check) |
|---|---|---|---|
| R0 | .515 (+.21) | .619 (+.48) | .577 (+.49) |
| top-4 | .409 (+.31) | .254 (+.16) | .223 (+.12) |
| **R1W** | .482 (-.15) | **.343 (-.04)** | .255 (-.02) |

**Go (R1W floor .343 <= .40).** The reverse pattern is better still (.255), so the gain is not a sign accident of
the chosen pattern. For R0 spreading hurts (.515 -> .58-.62).

### Addendum decision in cost terms (`results/cost_decision.txt`; projection)
T317e's DEV2 MSE x C is 3,435 pp^2 FPE, and the bar is .85 x 3,435 = 2,920.
- Arm B spread+ on R1W: projected RMS .720, C unchanged at 5,435, MSE x C 2,815 = .82 x T317e. **It passes, narrowly**
  (projected DEV2 F ~3.8 under the .074 convention).
- Spread- (not pre-registered for the decision): .74 x.
- (a) and (b): 2.4x and 4.0x WORSE than T317e (their added cost outweighs the gain).
- The smooth verdict: 3.6x worse.

## Status
Done. Unverified. The re-check was pushed (1d53520f) before smooth_verdict/ and Fable r5 §2 were read.

## Comparison with T319's own write-up (read after the re-check was pushed)
- **Definitions are identical:** mu and s^2 of N | < v3 (v4); the binding rival i* at the closure point; J =
  (A_j - A_i*) on the lows; the product over 9 rivals; paired oracle; 100k inputs per model. T319 computes A by
  forward mode, as I do. The only difference is seeds and code.
- **Agreement:**
  - R0 hard .532 vs .527, smooth 1.329 vs 1.326, product 11.69 vs 11.69;
  - top-4 hard .416 vs .399, smooth .662 vs .657, product 5.87 vs 5.88;
  - means within .01-.02 pp.
- **Independent check of T319's key failure: reproduced** (AGENTS rule 5). Together with T319's own exhaustion list,
  its two untried variants are now tried: cone-clipped J fixes the smoothing on R0 and top-4 but not on R1W, and the
  1-D walk gains 9% on R1W at 18k FPE. The Gaussian-smoothed closure family is negative for the deployable closure
  (R1W), scoped to these four forms (smooth, product, cone-clipped, 1-D largest low) at first order.

## Compute accounting
- Diagnostic sampling only; no estimator was built.
- `oracle.py`: 80 x 1e5 inputs, 3 closures; per input, a closure forward with 10-column tangents and a 1-D walk of
  ~10-12 cells. About 60 CPU-min.
- `spread.py`: 80 x 1e5 x 9 forward passes, ~3 CPU-min.
- Cost model for deployment: per pre-registration, at one point per triple (720 per model). Smooth +1.5 FPE, clip
  +8.5, product +13.5, walk +1.5 + 2 per cell; the spread pattern adds nothing (Fable r5: the prefix tree keeps its
  2,815 nodes, not re-verified here).

## Claims
1. [measured] T319's negative reproduces with independent code. The Gaussian-smoothed verdict Phi(m/(s||J||)) is 1.6-2.5x
   worse than the hard verdict (R0, top-4), and the product over rivals is 11-14x worse.
2. [measured] Cone-clipping J removes most of that damage where the closure keeps v4's position: top-4 .416 -> .195,
   R0 .532 -> .387. On R1W, the closure T317e uses, it makes things worse (.481 -> .553) and costs ~6k FPE.
3. [measured] Exact 1-D treatment of the largest-|J| low gives R1W .440 (-9%), top-4 .226, R0 .354, at ~15-19k FPE
   added. It fails both the go and the stop line on R1W, and the cost decision by 4x.
4. [measured] One position-deterministic spread pattern of the lows lowers R1W's floor .482 -> .343 (go <= .40), with
   bias -.15 -> -.04, at no added cost. The opposite pattern gives .255. Projected T317e MSE x C .82 x today
   (bar .85 x): a narrow pass.
5. [inferred] Spreading helps closures that keep v4 (top-4, R1W) and hurts R0. With v4 pinned, the spread lows mimic
   the true lows' variation, which the mean point removes. In R0 one value is shared by all 7 lows, and the pattern
   is a large deterministic shift.
6. [speculative] Since both signs help R1W and the reverse pattern helps more, the gain is from breaking the lows'
   equality, and the best single pattern is not yet known.

## Alternatives not ruled out
- The cost projection assumes the closure error adds in quadrature with T317e's other error terms and is unchanged by
  the walk integration. A real T317e run with the spread closure is the test (it changes the estimates, so it counts
  as a new variant).
- spread- beat spread+, but choosing it now is post hoc (forking paths). Only spread+ was pre-registered for the
  decision.
- 100k inputs per model: paired s.e. ~.05 pp, small against the lines. LB3 seeds and HP panels not run.
- (a)'s 64-node quantile rule and (b)'s lower cut at -8 are approximations (both checked: the walk matches MC, and the
  mass below -8 is < 1e-15).
- Not tried: second-order or two-low exact treatments; clipping in R1W with v4's own interval; spread patterns other
  than (-1)^j.

## Failure modes
- First-order laws of the lows through one gate pattern overstate the spread of the margin (T292, T319, here),
  unless they are clipped to the cone, and even clipped they fail when the closure moves v4 off its position (R1W).
- Exact 1-D walks are accurate but cost ~10-12 cells per point. A per-triple deployment already exceeds the budget.

# T339: T317f with the one-pattern spread closure (build; DRI lane decision 2026-10-18)

## Pre-registration (written before any T339 code ran)
**Base:** frozen T317f, copied verbatim from claude/t218-tie-tail-labels @88453669 (`cont_f_t317f.py` sha256
327dbb4a..., `estimator_f_t317f.py` 3afa3912..., both checked). **Only change:** the closure's low pattern. In T317f
the 6 lows (`LOWS`, every free position except the latest, which holds v4) share one value, low = mu(v4) linearised
along the walk (`low`, `dlow` in `inputs`). In T317g the low at absolute position j becomes

    low_j = low + s * (-1)^j * sigma,   dlow_j = dlow + s * (-1)^j * dsigma,

with sigma = sd(N(0,1) | < v4) and its derivative linearised along the walk exactly as T317f linearises mu:
sigma4 + dsigma4 (v4 - v4*) on the 2~3 line (v4 moves, d4 = -1), and the constant sigma4 on the 1~2 line (v4 fixed).
Here sigma^2(v) = 1 + v mu - mu^2 and dsigma/dv = (mu + v mu' - 2 mu mu') / (2 sigma), all model-free constants per
region, computed once at import. Nothing else changes: G, TMAX, screen C1, the walk, merge, profile, FLOP charges, and
the v4 node.

**Candidates:** s = +1 (pattern mu + (-1)^j sigma) and s = -1 (mu - (-1)^j sigma). **Selection rule:** the one with the
lower DEV2 MSE x C (MSE vs the official truths over all 80 models, C = mean FLOPs as executed / 5948), with one freeze.
It becomes T317g only if its MSE x C is below T317f's. DEV2 is development; HP5 is the claim.

**Checks (before the selection):**
1. s = 0 reproduces T317f bitwise (estimates and FLOPs) on all 80 DEV2.
2. B2 prefix tree: count the first-cell nodes per line for s = 0 and s = +-1 (expected unchanged, 2,815 per line, since
   at a given step all triples with a low there share one value, whatever the pattern).
3. C as executed for s = +-1 vs T317f (expected nearly equal; walks can cross a different number of cells, which is
   charged as executed).

**Report:** RMS, bias, C and F = RMS^2 max(C, 1k) / mean p(1-p) on 80 DEV2 for T317f (reproduced; DRI quotes F 3.14,
C 3,326) and both patterns.
**Expected (from the T327 oracle):** closure floor .48 -> .34 (s=+1) or .26 (s=-1), so RMS ~.83 -> ~.75 / ~.72 at the
same C, and F ~3.14 -> ~2.6 / ~2.4. The +.49 pp bias (from the other declared approximations) remains; the oracle
closure bias moved -.15 -> -.04 / -.02, so the total bias may rise by about +.1.
**Alternatives:** the gain could fail to carry over, because the walk integrates the closure along gap lines at
quadrature nodes, not at each input's true skeleton; that is what this run tests.
