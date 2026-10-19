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

## Status
Pre-registered (A1 amendment); s526 development running.

## Results
(none yet)

## Compute accounting
(to fill)

## Claims
(none yet)

## Alternatives not ruled out
See the pre-registration above.

## Failure modes
(to fill)
