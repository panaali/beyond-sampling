# T386: de-aliased deterministic rare-event gradient (shifted line grids)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · from [T372](https://github.com/panaali/clean-neurons/tree/claude/t372-rare-gradient/m16/approaches/rare_gradient)

## Question
T372 found that the exact weight-derivative of the deterministic K3 estimate of E1 = {argmax = rank-3 position,
g23 > t0} is wrong (2/12 directions within 2x on dev3_s528 + orig), although K3's *value* is right at nearby weights
and its finite difference at a step comparable to the line spacing matches the truth. T372's diagnosis: aliasing.
The estimate is a finite sum of line-island masses, the islands slide across fixed lines as W moves, and the
pointwise slope is dominated by the few lines that cut an island near its tip. **Does a fixed, deterministic
de-aliasing of the line rule (no random numbers) give a weight-gradient that passes T372's go line?**

## Method (pre-registered 2026-10-19, before any de-aliased gradient was computed on any model)

**Estimator.** The K3 estimate is a quadrature over (v2, g12) of exact line masses along g23 (T340c / T354 / T364).
v2 uses a 20-node Gauss-Legendre rule on each of 2 regions ([0, 6.5], [-6.5, 0]). g12 | v2 uses a 3-node Gauss rule.
Each (v2, g12) node carries 720 triple lines, and the closure (lows pinned) is model-free. The de-aliased estimator
keeps everything except the (v2, g12) node set. That set is replaced by **the average over a fixed set of shifted
grids**:
- v2 `mid20xM`: the average over M shifted 20-panel midpoint grids, nodes lo + h (j + (m + 1/2)/M), h = (hi - lo)/20,
  m = 0..M-1, weight h/M. The union is a 20M-node midpoint rule.
- g12 `qmid3xMg`: the average over Mg shifted 3-cell quantile grids of the exact law of g12 | v2 (density ∝ phi(v2 +
  g12), g12 >= 0), nodes at the (k + (m + 1/2)/Mg)/3 quantiles, weight 1/(3 Mg).
- The gradient is T372's exact boundary-term derivative of this estimate (`dealias.est_grad`, the same endpoint code).
  This is the analytic derivative of a line-spacing-averaged estimate. It is deductive: no random numbers, and the
  grids are model-free and fixed in advance.

`dealias.est_grad(model, t0, "gl20", ("gauss", 3))` reproduces T372's K3 estimate and 432-gradient bit for bit on
dev3_s526 (`python repro.py dev3_s526`: est 1.20996e-9, max gradient difference 0). The T372 helpers are copied
unchanged as `t372_*.py`, with only the import names and scratch paths changed. The frozen upstream code is asserted
by sha256 in `t372_k3src.py`.

**Development grid (dev3_s526 only, E1(3.25)).** M ∈ {1, 2, 4, 8, 16} × g12 ∈ {gauss3, qmid3x1, qmid3x2, qmid3x4,
qmid3x8}. Cost grows like M x (g12 nodes)/3 walks. Score each config's 6 directional derivatives g·u against
T372's s526 truths (`truth_fd2_dev3_s526.json`, FD at the level chosen by A1) with T372's rule. Also report
cos(g, g_lineCMC) against T372's s526 line-CMC gradient.

**Freeze rule (fixed now).** Let `finest` be (M = 16, qmid3x8). Freeze the **cheapest config (fewest lines)** with
(a) >= 5/6 s526 directions passing, and (b) cos(g, g_finest) >= 0.95. If several have equal cost, take the one with
the larger cos. If no config meets (a) and (b), freeze `finest` and score it anyway. The frozen config is written to
`results/frozen.json` and pushed **before** any de-aliased run on s528 or orig.

**Scoring (T372's go line, A2 form).** dev3_s528 E1(2.75) and orig E1(1.25), the 12 frozen T372 directions (5 random
+ top per model, `dirs_*.json`), against T372's truths (`truth_fd2_*.json`, FD_sel / FD_sel_se). A direction is
*resolved* if |FD| > 3 s.e. **GO** if at least 80% of the resolved directions have the same sign as the truth and
0.5 <= pred/FD <= 2, and at least 8 of 12 are resolved. Otherwise NO-GO. pred = g·u is the absolute derivative,
as in T372. Also reported, not gating: the value est/truth, cos with T372's line-CMC gradient, and the
T372 "converged FD" variant.

**Cost.** Executed FLOPs in FPE (5948 FLOPs). Report the walk cost and the endpoint backward-pass cost, each relative
to the single K3 walk (T372: 574k / 325k / 388k FPE for orig / s526 / s528).

**Class.** Deterministic (deductive: no random model inputs, fixed model-free lines). ARC acceptance unknown.

**Alternatives to keep in mind (rule 3).** (a) Even a perfectly de-aliased rule converges to the derivative of the
*closure-limited* estimand (lows pinned), not of P. K3's value bias (1.2-1.3x) may come with a derivative bias.
This is separated by comparing with the line-CMC gradient, which has no closure. A high cos with stable 2x failures
points to closure; a low cos to residual aliasing. (b) A pass on s526 could be tuning luck on 6 directions. This is
why s526 is dev only and why the freeze needs convergence (b), not only the dev score. (c) The midpoint/quantile
grids change the *value* rule as well. If the value moves far from the truth, a gradient pass may be coincidence.
The value is reported.

### Amendment A1 (2026-10-19, during s526 development; before any s528 or orig run of any T386 rule)
**What was seen first (s526 only).** 9 of the 25 family-A configs had run. Their top-direction slopes swing
136-558 (truth 298) up to 8x the lines, with no sign of settling. `scale.py` on the K3 endpoints measured the margin
change between adjacent lines: median 2.7 per v2 spacing and **61 per g12 spacing**, against |dm/dt| ~ 7 along the
line. So the error islands are thin bands in g12, and the g23-lines cross them almost tangentially. A g12 grid 8x
finer cannot resolve a band that thin.

**Smoothed indicator (family S, built, not pursued).** `smooth.py` is a smoothed boundary: S(m*/eps) with a biweight,
band integrated exactly per gate cell. Its eps -> 0 limit reproduces K3's value and gradient (cos 1.0000,
`check_smooth.py`). A bandwidth tied to the line spacing would be eps ~ 60 margin units, far larger than the islands
themselves, so the smoothing bias would dominate. Not pursued for scoring.

**Family T (added): transversal lines.** The same closure estimand as K3, with the lines turned to run along g12
(the top input) instead of g23: `transversal.py`. Given v2, g12 and g23 are independent. The g12 integral is exact
along each line (density phi(v2 + g12), g12 in [0, 8]). g23 gets a Kt-node Gauss rule for its density
8 Phi(v2 - t)^7 phi(v2 - t) on [t0, 6], and v2 uses dealias.v2_rule. The gradient is T372's boundary term with
dm/dg12. Checked: analytic directional derivative vs central FD (h 1e-6) of the T estimate on s526 top: 3.52355e-7
vs 3.52355e-7. **Seen before this amendment (s526 only, dev):** T with Kt = 3 / gl20: est 1.145e-9 (1.14x),
d log P per direction 22.6, 1.3, -0.6, 18.6, -45.3, 350 (truth 15.8, -0.82, -1.57, 13.5, -38.4, 298).
T dev grid: Kt ∈ {3, 5, 8} × v2 ∈ {gl20, mid20x2, mid20x4}. Its `finest` is (Kt 8, mid20x4).

**Freeze rule (replaces the one above; the family-A grid is still completed and reported).** Over both families,
freeze the config with the **lowest executed walk cost (FPE) on s526** among those with (a) >= 5/6 s526 directions
passing and (b) cos(g, g_finest of its own family) >= 0.95. If none qualifies, freeze the config with the most s526
passes among the configs with (b). Break ties by the higher cos with T372's s526 line-CMC gradient. If no config has
(b) either, freeze the finest T config. Scoring, go line, models and directions are unchanged. Exactly one config
is scored.

### Freeze (2026-10-19, before any s528 or orig run)
`python dev_score.py --freeze` -> `results/dev_s526.txt`, `results/frozen.json`. 26 configs ran (all 9 T, 17 of 25 A).
The 8 remaining A configs (M 2-16 with qmid3x2 / x4 / x8 and mid20x2/gauss3) were stopped. Every one of them costs
more than the pick, so none could change it. **The A1 rule freezes `mid20x1 / qmid3x1` (family A, 1.0x the K3 walk:
5/6 s526 passes, cos .996 to A's finest).** This config is the gating one and is scored exactly as pre-registered.
Caveat, written before scoring: on s526 the pick looks like luck rather than convergence. Its 2x-lines neighbour
mid20x2/qmid3x1 passes 1/6, and cos >= .95 turned out to be a weak criterion, because every config has cos >= .91
(one dominant gradient direction). **Secondary, declared now and not gating:** `T5 / gl20`, the cheapest T config
meeting (a)+(b), and `T8 / mid20x4`, the converged T. The converged T separates residual aliasing from closure
bias. On s526 the two families' finest configs agree: d log P along the 6 directions is 21.4, 1.3, -1.5, 17.4,
-43.5, 333.6 (A, 128x the lines) vs 21.2, 1.5, -1.7, 17.2, -42.6, 328.4 (T); values are per truth P.

## Status
**Done.** Gating (pre-registered A1 freeze, `mid20x1 / qmid3x1`, family A): **NO-GO**, 6/12 resolved directions pass
on s528 + orig (go needs >= 80%). Secondary, declared before scoring and not gating: transversal lines (family T)
`T5 / gl20` **10/12** and `T8 / mid20x4` **11/12**. Both are at GO level, at 2.8-3.7x and 18-24x the K3 walk cost (total FPE).

## Results
All from `python score.py` (-> `results/score.txt`, `results/score_t386.json`), on `results/cfg_<model>_<cfg>.json`
(`python run_cfg.py <model> <cfg>`). Development table: `python dev_score.py` (-> `results/dev_s526.txt`).
Values are d log P per unit step along u, i.e. (g . u) / P_truth; the truths are T372's (FD at the A1 level).

| model | dir | truth | K3 (T372) | **gating** mid20x1/qmid3x1 | T5/gl20 (sec.) | T8/mid20x4 (sec.) | line-CMC (T372) |
|---|---|---|---|---|---|---|---|
| orig | 0 | 40.8 ± 2.6 | -5.3 ✗ | -4.1 ✗ | 30.8 ✓ | 32.1 ✓ | 38.7 |
| | 1 | -46.8 ± 2.7 | -2.3 ✗ | -2.3 ✗ | -23.2 ✗ (0.496) | -29.7 ✓ | -43.1 |
| | 2 | 17.6 ± 1.6 | 8.8 ✗ | 29.9 ✓ | 20.3 ✓ | 37.7 ✗ (2.14) | 21.6 |
| | 3 | -34.0 ± 1.9 | -0.2 ✗ | -4.8 ✗ | -26.1 ✓ | -26.6 ✓ | -31.0 |
| | 4 | -28.5 ± 1.3 | -7.9 ✗ | -31.0 ✓ | -37.4 ✓ | -39.5 ✓ | -27.6 |
| | top | 390 ± 26 | 217 ✓ | 615 ✓ | 553 ✓ | 625 ✓ | 411 |
| s528 | 0 | -17.5 ± 0.7 | 4.0 ✗ | -10.4 ✓ | -22.6 ✓ | -23.6 ✓ | -18.4 |
| | 1 | 3.4 ± 0.3 | 11.7 ✗ | 26.4 ✗ | 9.0 ✗ (2.68) | 3.7 ✓ | 0.3 |
| | 2 | 20.2 ± 0.5 | 7.1 ✗ | 25.5 ✓ | 30.6 ✓ | 28.1 ✓ | 22.3 |
| | 3 | 24.3 ± 0.5 | 7.8 ✗ | 37.4 ✓ | 38.9 ✓ | 34.7 ✓ | 26.1 |
| | 4 | -12.2 ± 0.4 | 13.8 ✗ | 24.1 ✗ | -13.7 ✓ | -17.8 ✓ | -17.5 |
| | top | 423 ± 11 | 454 ✓ | 907 ✗ (2.15) | 719 ✓ | 597 ✓ | 404 |
| **pass (s528 + orig)** | | | 2/12 | **6/12 NO-GO** | 10/12 | 11/12 | 11/12 |
| s526 (dev) | 0..top | 15.8, -0.82, -1.57, 13.5, -38.4, 298 | 1/6 | 5/6 | 5/6 | 5/6 | 5/6 |

- **Values [measured]** (est / truth P): gating 1.12x (orig), 1.33x (s528), 1.11x (s526). T5: 1.24x / 1.40x / 1.14x.
  T8: 1.45x / 1.47x / 1.14x. T's value rises with refinement on orig and s528. The closure estimand's continuum
  limit lies above P (H-R1 within-2x scope still holds), and T's directional derivatives are about that much too
  large (the ratios are mostly 1.1-1.6). That is the closure bias of alternative (a), not aliasing.
- **cos(g, g_lineCMC) [measured]:** K3 .63 / .82 / .97 (orig / s528 / s526), gating .64 / .85 / .99, T5 .84 / .99 /
  1.00, T8 .87 / 1.00 / 1.00.
- **Convergence [measured, s526]:** the finest A (128x the K3 lines, 41M FPE) and the finest T agree per direction
  to <= 0.2 in d log P (21.4 / 21.2, ..., top 333.6 / 328.4). The two orientations converge to the same derivative.
  K3-direction (family A) configs below 16x the lines swing 128-615 on the top direction. T configs at any size
  stay at 325-350.
- Against T372's not-pre-registered "converged FD": gating 6/11, T5 9/11, T8 9/11.

## Compute accounting
FPE = 5948 FLOPs, executed (walk FLOPs as T354/rare.walk3 count them, plus one forward+backward per endpoint,
`t372_rnn.flops_per_grad`). Model-free line construction is not counted. All 432 derivatives come from one run.

| config | orig | s528 | s526 |
|---|---|---|---|
| K3 walk (T372, reference) | 574k | 388k | 325k |
| gating mid20x1/qmid3x1 | 616k + 4.6k (1.1x) | 405k + 3.3k (1.0x) | 317k + 1.5k (1.0x) |
| T5/gl20 | 1.52M + 70k (2.8x) | 1.36M + 86k (3.7x) | 1.09M + 43k (3.5x) |
| T8/mid20x4 | 9.65M + 441k (17.6x) | 8.61M + 541k (23.6x) | 7.21M + 309k (23.1x) |

(x = total relative to the K3 walk.) For comparison, T372's line-CMC used 33-58M FPE for 5e6 lines. Development on
s526: 26 configs, about 80M FPE in total (dominated by the 128x A config).

## Claims
- [measured] **Pre-registered verdict: NO-GO.** The config frozen by the A1 rule (one unshifted midpoint/quantile
  grid, K3's cost) passes 6/12 on s528 + orig. Scope: family A shifted-grid averages along g23-lines on these 3
  models and thresholds, frozen on 6 dev directions by a cos >= .95 convergence criterion. That criterion proved too
  weak: every config had cos >= .91.
- [measured] The K3 aliasing has a measurable cause. At K3's island endpoints the argmax margin changes ~61 units
  between adjacent g12 lines and ~2.7 between adjacent v2 lines, against ~7 per unit g23 along the line
  (`scale.py`, s526). The islands are thin bands in g12 that the g23-lines graze.
- [measured] Turning the lines to run along g12 (family T, same closure estimand, deterministic) gives a stable
  gradient. T5/gl20 (2.8-3.7x the K3 walk) passes 10/12 and T8/mid20x4 passes 11/12 on s528 + orig. These are
  **secondary, declared before scoring but not gating**; one agent, one implementation, unverified.
- [measured] On s526 the shifted K3 grids converge, at 128x the lines, to the same derivative as T. So the K3
  failure is quadrature (aliasing), not the boundary formula.
- [inferred] T's remaining error (ratios 1.1-1.6, value 1.14-1.47x) is the closure bias of the K3 estimand, not
  aliasing. It tracks the value bias, which grows with refinement on orig / s528.

## Exhaustion checklist (for the gating negative)
- Tried: K3 lines with shifted v2 midpoint grids (M = 1, 2, 4, 8, 16) × g12 Gauss-3 or shifted quantile grids
  (Mg = 1, 2, 4, 8): 17 of 25 dev configs on s526; 8 costlier configs stopped once the pick was fixed.
- Tried, not scored: smoothed indicator (`smooth.py`, exact band integration, eps -> 0 check passes). The
  line-spacing-tied bandwidth would be ~60 margin units, larger than the islands.
- Tried, scored secondary: transversal lines (family T), GO-level.
- Not tried: a stricter freeze criterion (e.g. per-direction stability across 2x refinements) for family A; adaptive
  refinement around hit lines; a closure fix (T369) to remove the value / derivative bias of T; T on
  triple-tie-dominated models; more models.

## Alternatives not ruled out
- T's pass could partly reflect T372's truth choices. s528 dir 1 (truth 3.4 at A1 level 0; converged -0.35) is
  curvature-affected, and T8 passes it, but at the converged FD it would not.
- 12 directions on 2 models is a small test, and T was added after dev data was seen (A1, s526 only). An
  independent re-check on new models / thresholds with T frozen as is (T5/gl20) is the natural next step.
- T's magnitudes run high with the value bias. A model where the closure bias exceeds 2x would fail T too.

## Failure modes
- Fixed lines nearly parallel to thin islands: the value can look fine while the derivative aliases (K3,
  family A). Check the margin change between adjacent lines vs along the line (`scale.py`) before trusting a
  quadrature gradient.
- A selection rule based on cos with a converged gradient is weak when one direction dominates the gradient.
- T inherits K3's closure. Its value and derivative drift upward with refinement on orig / s528 (1.45-1.47x at T8).
