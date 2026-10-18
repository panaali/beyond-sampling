# budget_t317d: T324, per-approximation error budget and deductive-status audit of the T317d candidate

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · candidate: claude/t218-tie-tail-labels @5b4d4806
m16/approaches/r1_cont/ (cont_d.py, estimator_d.py) · red-team: review/candidate_redteam/README.md (ccr)

I am a different agent from the T241/T218 child that built the candidate and did the T312/T320 checks. I read the
candidate's code (needed for both parts) but wrote all the budget code myself.

## Pre-registration (written 2026-10-18, before any T324 number)
**Pointwise form of the estimator** (derived from reading r1w.est / cont_d.est). Per ordered top-3 triple and region
r (nb: v2 > 0, b: v2 <= 0), with r's transverse point (v2*, g12*, g23*) (cont_d.TABLES, exact law):
- the line-23 verdict s23_r(t) = 1[model correct at the line-23 input]. The input has v1 = v2* + g12*, v2*, v3 = v2* - t,
  and v4 at the latest free position = v3 - delta (delta = v3* - v4*, v4* = gauss_v4(v3*, 1)), with the 6 lows at
  mu(v4*) + mu'(v4*)(v4 - v4*). s23 = 1 for t >= tmax = 0.5.
- the line-12 verdict s12_r(t) at v1 = v2* + t, (v2*, v3*, v4*), with the lows at mu(v4*). s12 = 1 for t >= tmax, and
  s12 = 1 in region b ("blind g12 assumed correct").
- est = E_x[s23_r(g23(x)) s12_r(g12(x))] over the exact input law, up to G = 20 Gauss-Legendre in v2.
  **Check (must pass before any budget number is used):** the MC of this pointwise form must agree with cont_d.est
  within 4 combined SE on every model, and its RMS difference must be <= 2x the MC SE.

**Switch-offs** (each replaces one approximation by its exact counterpart, evaluated pointwise on the same x,
paired). For each model: 2e5 inputs, seed 324000 + index. Truths: the DEV2 1e8 open truths.
1. **tmax off:** no cut (lines evaluated at any t).
2. **blind g12 off:** region b uses its own walked line-12 verdict s12_b.
3. **mu linearisation off:** line 23 uses the exact closure-c law along t: v4 = gauss_v4(v3(t), 1), lows = mu(v4).
   This removes both the constant-delta v4 and the linear lows.
4. **product off:** joint 2-D verdict at (v2*, g12, g23) with line 23's closure, instead of s12(g12) s23(g23). The tmax
   cut and the blind rule are kept: beyond tmax in one gap, the other line alone; in region b, g12 is free.
5. **one transverse point off** (the v2 direction): both lines re-anchored at the input's own v2, at (v2, g12*, g23*),
   with the closure parameters recomputed there. The cross-gap part of "one transverse point" is the product
   approximation (4): with both coordinates at the input's own point, the lines meet at x and the product is the joint
   verdict.
6. **closure c off:** the true model on x (truth), measured as the last step of the path below.
- **Cumulative path** V1 -> 1 -> 1+2 -> 1+2+3 -> 1+2+3+4 -> 1+...+5 = E1 (closure c, exact law, MC) -> truth. The last
  path point must equal a direct MC of closure c pointwise (identity check).
- **Reported:** bias, RMS and MSE over the 80 models for every variant. Each approximation's share of the bias
  (+.44 pp) and of the MSE, both one-at-a-time (from V1) and along the path. The red-team's claim (about 80% of the
  error and all of the bias come from approximations 1-5) is tested as: MSE(V1) - MSE(E1) >= 0.6 MSE(V1), and
  |bias(E1)| < |bias(V1)| / 2. "Supported" iff both hold.

**Audit (part 2):** read every runtime file of estimator_d's import closure (cont_d, cont_c, cont, r1w, r1w2, walk,
v1b2, r0). List any random draw, any data-dependent extrapolation from observed model behaviour, and any constant
chosen on model data, with file:line.
Labels: the budget is diagnostic sampling. Cap 2.5 h.
- **Amendment (after the main run, before reporting; labelled post hoc).** The run exposed a sixth approximation
  that the pre-registered list folded into "closure c": R1W/T317d place v4 at the **single Gauss node** E[v4 | v3]
  (`r1w.gauss_v4(v3, 1)`, K = 1). T304 Stage A's closure c, the ".47 pp floor", uses the **true** v4 value
  (`floors.py:closure_inputs`). `e0.py` adds the step E1 (node v4) -> E0 (true v4) on the same paired inputs. The
  candidate README does name "a single v4 node" among its approximations.

## Method
- `budget.py` (own code): pointwise verdicts on 2e5 exact-law inputs per model, all variants paired on the same inputs.
  - The candidate is imported only for its transverse points / v4* (`cont_d.TABLES`) and for `cont_d.est` (the
    check).
  - `node(v3)` reproduces the candidate's `gauss_v4(v3, 1)` to 1e-15 (tabulated on its own grid rule).
  - The candidate's code needs `r0_integral/data/`. I ran it from a scratch copy at 5b4d4806, with DEV2 weights and
    truths placed there.
- `e0.py`: the true-v4 closure c on the same inputs.
- `analyze.py` -> `summary.json` (patched with E0 by the inline step recorded in this commit).
- Run: `python budget.py && python e0.py && python analyze.py` (~25 CPU-min).

**Checks [measured]:**
- The pointwise MC of V1 vs `cont_d.est` on all 80 models: max |z| 2.51, RMS difference .056 pp against a mean MC SE of
  .058 pp: **passed**. Bias +.439 vs +.444, RMS .796 vs .794.
- The path end equals a direct closure-c MC exactly (max |diff| 0).
- E0 reproduces T304 Stage A (bias -.154, RMS .463 vs -.156, .471).
- Not pre-registered, a red-team minor item: `walk_cont(max_cells=400)` is never hit on DEV2 (max 22 cells per line;
  the estimates are identical with a cap of 5000).

## Results (80 DEV2 retrains, 1e8 truths; pp)
**Cumulative path** (each row switches one more approximation off; MSE in pp^2, V1 MSE = .633):

| step | bias | RMS | change in bias | share of V1's bias | share of V1's MSE |
|---|---|---|---|---|---|
| V1 = T317d (pointwise) | **+.439** | **.796** | | | |
| 1. tmax cut off | +.256 | .720 | -.182 | 42% | 18% |
| 2. + blind g12 walked | +.159 | .685 | -.097 | 22% | 8% |
| 3. + mu linearisation off (exact closure along line 23) | +.177 | .691 | +.018 | -4% | -1% |
| 4. + product off (joint 2-D set at v2*) | -.154 | .583 | -.332 | 76% | 22% |
| 5. + transverse v2 off (= E1: closure c with node v4, exact integral) | +.155 | .475 | +.310 | -71% | 18% |
| 6. + single v4 node off (= E0: closure c, true v4; T304's floor) | -.154 | .463 | -.309 | 70% | 2% |
| truth (closure c itself) | 0 | 0 | +.154 | -35% | **34%** |

**One at a time from V1** (only that approximation off):

| switched off | bias | RMS | change in bias | change in MSE |
|---|---|---|---|---|
| tmax cut | +.256 | .720 | -.182 | -.115 (-18%) |
| blind g12 | +.370 | .759 | -.069 | -.056 (-9%) |
| mu linearisation | +.455 | .803 | +.017 | +.012 (+2%) |
| **product of line sets** | **-.129** | **.616** | **-.568** | **-.253 (-40%)** |
| transverse v2 (alone) | +.708 | .890 | +.269 | +.159 (+25%) |

**Red-team claim** ("about 80% of the error and all of the bias come from the declared approximations other than
closure c"):
- **Bias: supported.** The walk approximations together carry +.59 pp, more than V1's whole +.44. The closure itself
  has -.15 (true v4), so the two partly cancel (-.15 + .59 = +.44).
- **Error: partly.** The walk approximations (1-6) carry **66%** of V1's MSE (64% excluding the v4 node). The
  pre-registered >= 60% test passes, but 80% does not; the closure itself carries 34%. In RMS terms, .80 -> .46.
- **Pre-registered verdict: supported** (both pre-registered tests hold). The "80%" figure should read "about two thirds".

Which part carries it [measured]:
- **The product of the two 1-D correct sets** is the largest single item. It is responsible for -.57 pp of bias one
  at a time and 40% of the MSE; it overstates accuracy because a point that is correct on both central slices is not
  always correct jointly.
- **The tmax cut** is next (-.18 pp, 18% MSE): errors at gaps >= .5 are assumed away (the ANAT soft tail).
- **The blind g12 rule** carries -.07 to -.10 pp and 8-9%.
- **The mu linearisation** is negligible (|.02| pp).
- **The single transverse v2 and the single v4 node interact with the product** and largely cancel it:
  - re-anchoring the lines at each input's own v2 *alone* makes V1 worse (+.27 pp);
  - after the product is removed, re-anchoring removes 18% of the MSE.
  - The v4 node shifts the bias by +.31 pp but moves the MSE little.
  So the order-independent statement is about the group "product + transverse + v4 node": together they carry
  +.03 pp of bias (path 4-6 sum: -.33 +.31 -.31; one-at-a-time they are much larger and opposite in sign) and
  ~42% of the MSE. Their individual sizes depend on the order of removal.

## Part 2: deductive-status audit (estimator_d's import closure at 5b4d4806)
Files read: estimator_d.py, cont_d.py, cont_c.py, cont.py, r1w.py, r1w2.py, r0_walk/walk.py, r1_v1b2/v1b2.py,
r0_integral/r0.py.
1. **Random draws: none touch the model at runtime.** `est` uses only constructed inputs (`r1w.inputs`), exact walks
   and model-free tables. **But two seeded Monte Carlos still execute at import:**
   - `r0_walk/walk.py:32` (`TP = _transverse()`, 1e6 x 10 normals, `default_rng(0)`);
   - `r1_walk/r1w2.py:23` (`REG = {False: _regions(False), True: _regions(True)}`, twice 1e6 x 10).
   They run because cont.py imports walk and r1w2. T317d never reads them (cont_d uses its own exact TABLES; `REG`
   only in the `points="mc"` check; `TP` not at all). The estimates do not depend on them, but the red-team's
   "a reader who greps for random finds it" still applies to T317d's import closure. Fix: guard both behind a lazy
   function.
2. **Extrapolation from observed model behaviour: yes, structurally, and it is declared.** The model is evaluated
   exactly along two lines per triple and region. Those verdicts are then applied to the whole region:
   - every v2 in the region (one transverse point);
   - every value of the other gap (the product);
   - gaps beyond 0.5 (assumed correct);
   - blind g12 (assumed correct).
   That is extrapolating the model's verdict from one line to a 3-D region. It is the same for every model; nothing
   is fitted per model at runtime. These are the approximations measured above; the product and the transverse
   point are the largest.
3. **Constants chosen on model data (all on DEV2 by sampling):**
   - closure c among four candidates (T304 Stage A, 200k paired skeletons per model);
   - v4 at the latest free position (part of c);
   - TMAX = 0.5 (T302 scan of tmax 0.05-6 on DEV2, chosen for cost at small loss);
   - K = 1 v4 node and the blind-g12 rule (chosen by cost; their effect measured by MC, T304 V3);
   - G = 20 (a numerical convergence check on DEV2; it moves estimates by <= 1.4e-6 pp, so it is not an accuracy
     choice).
   Numerical constants (eps 1e-10, tie tolerances 1e-12, max_cells 400 never hit, v2 range +-6.5, the region cut
   at v2 = 0) are model-free.
   **No constant is fitted per model, and no runtime step uses random inputs.**
   **Conclusion:** "sampling-free at runtime; the approximation scheme was selected and validated by Monte Carlo
   on development retrains" (the red-team's wording) is accurate. "Deductive" in Hilton's stricter sense is not.

## Claims
1. [measured] Of T317d's DEV2 error (RMS .796, bias +.439, MSE .633 pp^2), the walk-level approximations carry
   **66% of the MSE and +.59 pp of bias**. Closure c itself carries 34% of the MSE and -.15 pp, so the net bias is a
   partial cancellation.
2. [measured] The largest single approximation is the product of line sets (one at a time: -.57 pp bias, -40% MSE).
   Next come tmax (-.18, -18%) and blind g12 (-.07, -9%). The mu linearisation is negligible.
3. [measured] The transverse v2 point, the product and the single v4 node interact strongly (their individual
   effects depend on the order of removal and partly cancel). Only their sum (~+.03 pp bias, ~42% MSE) is
   order-free.
4. [measured] Audit: no runtime random draws on the model. Two unused seeded model-free MCs execute at import. The
   design constants (closure choice, tmax, K, blind rule) were chosen by DEV2 sampling. No per-model fitting.

## Alternatives not ruled out
- Shares along a path depend on the order. I report one path plus one-at-a-time; other orders would move the
  product/transverse/v4 split, not the totals.
- DEV2 only (development models). The same budget on a sealed panel needs the scorer.
- The "exact counterpart" of the product uses line 23's closure in 2-D at v2*. Another 2-D closure convention would
  move step 4 vs 5, not E1.

## Compute accounting
~25 CPU-min of diagnostic sampling (80 models x 2e5 inputs x ~25 forward evaluations), plus 80 `cont_d.est` calls.

---

## Addendum A (Fable r5 D3): combination rules for the 2-D residue. Pre-registered 2026-10-18, before any addendum number.
Written after the main budget (above) was known. The T306-B mechanism hypothesis below is based on that budget's
one-at-a-time result.

**Target.** Per input x, T(x) = the closure-c verdict at x's own point (v2, g12, g23), with the K = 1 v4 node and exact
lows along t, keeping V1's tmax cut and blind-g12 rule. The **residue** of a rule X is X - T. It isolates
"transverse point + line-set product + mu linearisation", the approximations a combination rule can change.
**Models:** the lineage's 10 DEV2 models (indices 0, 8, ..., 72), 1e6 exact-law inputs each (seed 324500 + index).
**Rules** (all from verdicts on lines the estimator walks):
- (i) **P1:** the product s23 * s12 at the one transverse point per region (= V1).
- (ii) **P2:** the product at each half's own point (T306-B: v2 split at the region's median, exact-law points per
  half). Costs 2x the walks.
- (iii) zero-cost alternatives at the one point:
  - **OR** (union: correct if either line says correct);
  - **L23** (line 23 alone, g12 ignored);
  - **L12** (line 12 alone);
  - **AVG** ((s23 + s12) / 2).
  (min = AND = P1 for 0/1 verdicts.)
- (iv) **J1:** the exact 2-D set at the one point (joint verdict at (v2*, g12, g23), line-23 closure); **J2:** the same at
  the half points.
**Metrics:**
- model-level residue: mean over x of X - T, per model; bias and RMS over the 10 models;
- class-level residue: per (model, triple) mass difference, RMS over 7,200 classes (paired, MC SE reported).
**Go:** some zero-cost rule (iii) has model-level residue RMS <= 0.60 x that of P1. **Stop:** no zero-cost rule does, and only
J1/J2 get <= 0.60 x. Otherwise (nothing gets there): stop too, and report.
**T306-B anomaly, hypothesis to test:** refining the transverse point removes an error of the opposite sign to the
product's, unmasking it. Predictions: (a) P2's bias exceeds P1's, on the side of "transverse off alone" (+.71 on 80
models); (b) J2's bias is closer to 0 than J1's (with the product gone, refinement converges). If (b) fails, the
hypothesis is wrong.
- **Amendment A1 (before any addendum number).** V1's tmax cut and blind rule are not separable at an input's own
  point (both lines then pass through x). So the target and every rule are evaluated with **no tmax cut and blind g12
  walked** (the budget's path-2 state). T(x) = closure c (node v4, exact lows) at x. The rules use the estimator's
  linearised line-23 closure (mu-lin on) unless marked "exact". The P1 residue is then V1's path-2 point minus E1.
