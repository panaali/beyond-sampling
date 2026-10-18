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

### Addendum A results (`python addendum.py`, `addendum.json`, `addendum.log`; 10 DEV2 models x 1e6 inputs, ~25 CPU-min)
Residue = rule - T (T = closure c at x's own point; no tmax cut, blind walked). Model-level stats are over 10 models; class
level is the per (model, triple) mass difference (about 1,390 inputs per class, so the class-level MC noise is ~.6-.9 pp).

| rule | cost | residue bias | **residue RMS** | ratio to P1 | class-level RMS | vs truth: bias / RMS |
|---|---|---|---|---|---|---|
| (i) P1: product at one point (V1) | 1x | -.045 | **.385** | 1.00 | 4.9 | +.109 / .600 |
| (iii) OR (union) | 1x | +6.53 | 6.84 | 17.7 | 9.0 | |
| (iii) L23 (line 23 alone) | 1x | +2.00 | 2.24 | 5.8 | 5.4 | |
| (iii) L12 (line 12 alone) | 1x | +4.49 | 4.65 | 12.1 | 7.2 | |
| (iii) AVG | 1x | +3.24 | 3.41 | 8.9 | 4.9 | |
| (ii) P2: product at the half points (T306-B) | 2x walks | **+.334** | .495 | 1.28 | 3.6 | +.489 / .741 |
| (ii) OR2 | 2x | +6.54 | 6.82 | 17.7 | 8.9 | |
| (iv) J1: exact 2-D set at one point | 2-D | -.345 | .377 | .98 | 2.0 | -.191 / .602 |
| (iv) J1, exact closure | 2-D | -.308 | .343 | .89 | 1.7 | |
| (iv) **J2: exact 2-D set at the half points** | 2-D, 2x | -.008 | **.087** | **.23** | 1.0 | +.147 / .526 |
| target T (closure c) | | 0 | 0 | | | +.154 / .474 |

**Verdict (pre-registered): STOP, only the exact 2-D set works.** Every zero-cost rule is 6-18x *worse* than the
product. min = AND is the product itself. The two line verdicts carry separate error sets, and anything looser than
their intersection counts the other line's errors as correct. Only J2 (exact 2-D sets, two anchors per region) cuts
the residue by >= 40% (by 77%). J1 alone does not at model level (.98), although it is 2.5x better per class.

**T306-B anomaly explained (both pre-registered predictions hold) [measured]:**
- (a) P2's bias exceeds P1's: -.045 -> +.334 (a shift of +.38; T306-B saw +.45 -> +.78, +.33).
- (b) With the product gone, refining the anchor converges: J1 -.345 -> J2 -.008.
- So the residue is two errors of opposite sign that cancel at one anchor:
  - **the product overstates accuracy by about +.30-.34 pp** (P1 - J1 = +.30; P2 - J2 = +.34);
  - **the single anchor understates it by about -.34 pp** (J1 - J2 = -.34).
- Refining only the anchor (T306-B) removes the negative part and exposes the product's positive part. The rise is
  what a two-error system does, not non-convergence.
- **The residue's model-level RMS (.385) is small because of this cancellation.** Per class it is 4.9 pp (J2: 1.0).

**Reading for the decision.** The "nearly free" fix does not exist among combination rules. Removing the residue needs
the exact 2-D correct set (a 2-D cell enumeration per triple, not a line walk) at >= 2 anchors per region. At the
paradigm's ~1:1 cost trade that doubles-plus the walk cost. Removing one of the two cancelling errors alone makes the
estimator worse.

---

## Addendum B (T333). Pre-registered 2026-10-18 after Addendum A's results, before any B number.
Addendum A already answers T333's (i)-(iv) and the T306-B anomaly in the residue frame. B adds three things: the DRI's
go metric in V1's own convention; a geometry-derived zero-cost rule; and the executed cost of the exact joint set.

**B1. Recovery of the product's MSE share.** On the same 10 DEV2 models (indices 0, 8, ..., 72; budget.py's paired
2e5 inputs, seed 324000 + index), in V1's convention (tmax cut and blind rule kept), all against the 1e8 truth:
- P1 = V1;
- J1 = the joint 2-D set at v2* (the budget's off_product);
- rule X's recovery = (MSE(P1) - MSE(X)) / (MSE(P1) - MSE(J1)).
Zero-cost rules: OR, L23, L12 (from A), and the new **ADD** (additive margin from the crossing). With L23(t), L12(u)
the logits along the two lines and L0 the logits at their crossing (v2*, g12*, g23*), which both lines contain:
L_add = L23(g23) + L12(g12) - L0, and correct iff argmax L_add = label. ADD is the first-order (cut-HDMR) joint
margin. It uses only the logits the walks already compute, so it adds no network evaluation; its correct set is
bounded by piecewise-linear curves. The same cut conventions as J1 apply.
**Go:** some zero-cost rule recovers >= 40%. If so, rerun it on all 80.

**B2. Cost of the exact joint set (2-D cell enumeration at one v2 per triple).**
- Census: on 3 DEV2 models (0, 40, 72), count distinct gate patterns (all 10 x 16 gates) of the joint-set input
  (line-23 closure) on a 100 x 100 grid over (g12, g23) in [0, tmax]^2. Do this per triple, for the nb and b regions,
  plus 200 x 200 on model 0 as a convergence check. The count is a lower bound on the number of 2-D linear cells.
- Cost per cell, by the written convention (Directive 8; cont.py / v1b2 constants):
  - an affine form in 2 directions per step: values + 2 tangents, 3 x (2H^2 + 2H) per step x 10;
  - edge lines from all 160 pre-activations: about 160 half-plane tests per cell vertex set (counted
    160 x 6 vertices x 4);
  - readout with 2 tangents: 3 x 2CH;
  - the correct polygon: 9 rival half-planes x 6 x 4;
  - the Gaussian mass of the correct polygon per GL v2 node: about 6 edges x 2 Phi x 20 FLOPs x G = 20.
  FPE = cells x per-cell FLOPs / 5948, compared with T317d's ~5.5k.
Labels: diagnostic sampling (the census and B1). Cap 1.5 h.

### Addendum B2' (DRI refinement, 12:24): J2's executed cost. Counting rule pre-registered 2026-10-18 before any count.
It replaces B2's 3-model census, which I stopped before it printed anything.
- **What is counted.** For each of the 10 models (0, 8, ..., 72), each of the 4 half anchors (nb1, nb2, b1, b2; exact-law
  points as in Addendum A) and each of the 720 triples: the number of distinct full gate patterns (10 x 16 bits) of the
  joint-set input (line-23 closure, as J2) on a 100 x 100 midpoint grid over (g12, g23) in [0, tmax]^2 = [0, .5]^2.
  This is the walked window. Beyond it, V1's tmax convention applies.
  - Convergence: one model (0) at 200 x 200.
  - The count is a **lower bound** on the number of 2-D linear cells: cells smaller than the grid are missed, and a
    pattern split into disconnected pieces counts once.
- **Pricing.** Under the DRI's line-cell convention: 0.54 FPE per cell after the first, plus a first cell per
  (triple, anchor) at 2.0 FPE (one value + tangent pass, without prefix sharing). Alternative (my B2 per-cell
  convention for a 2-D cell): 4.39 FPE per cell. Both are reported. The 2-D price is the honest one for an
  enumeration with two tangents, half-plane clipping and polygon masses; the line price is a floor.
- **J2's C** = (sum of cells) x price + T317e's other costs (left unchanged). Two cases:
  - (a) V1 convention: the blind region keeps its line-23 walks at the 2 b half anchors (g12 assumed correct), so it
    has no 2-D cells;
  - (b) blind walked in 2-D.
  I also report the extra cost beyond T317e's lines (~5.5k) and T330-C1's 3.3k.
- **J2's RMS vs truth**, closure c remaining, on the same 10 models (budget.py's paired inputs): (a) V1's tmax and blind
  conventions with J2 in place of product + transverse; (b) no cut and blind walked (Addendum A's J2: .526).
- **Strict arithmetic (DRI):** at the closure-c floor, strict needs C <= ~3.45k FPE all-in. If J2's extra cells exceed
  ~1.2k FPE beyond C1's 3.3k, strict is out of reach for closure c by arithmetic.

### Addendum B3 (DRI 12:51/12:52): where the product's overstatement lives, and a corner-only fix. Pre-registered 2026-10-18 before looking.
- **Data:** the same 10 models and budget.py's paired 2e5 inputs, in V1's convention (tmax cut, blind rule). The
  closure is the mean-point closure c (node v4, linearised lows on line 23), **not** T339's spread-lows closure. The
  geometry of the product error may differ under spread lows.
- **Product error per input:** D = P1 - J1 (one anchor), and also D2 = P2 - J2 (half anchors). Over = (D > 0) mass,
  under = (D < 0) mass, net = their difference, all as pp of all inputs.
- **Location:** by the input's own gaps. Delta grid {.025, .05, .1, .2, .5}. For each delta, the share of over / net mass
  in the corner (g12 < delta and g23 < delta), and in the L-band (min(g12, g23) < delta).
- **Model dependence:** per model, the over and net mass against 1 - p (truth), as an OLS slope and correlation. Is it
  near-constant or does it scale?
- **Corner fix:** C_delta = J1 inside the corner, P1 elsewhere (and the J2/P2 analogue). Bias removed = mean(P1 - C_delta)
  in pp. Cost = the cells of the joint set inside the corner [0, delta]^2 per triple and nb anchor (blind region: the
  product does not apply, since g12 is assumed correct), by the B2' census rule on a 50 x 50 grid over the corner,
  priced at 0.54 FPE per cell beyond the first (the first cell is already paid by the lines).
  - **Go:** some delta removes >= .25 pp of bias at <= +600 FPE.

### Addendum B results (T333)
**B1 (`python addB.py b1`, `addB1.json`; 10 DEV2 models; V1 convention, vs truth). No go.**

| rule | bias | RMS | MSE | recovery of the product's MSE share |
|---|---|---|---|---|
| P1 = V1 | +.341 | .729 | .531 | 0 |
| J1 (joint set at v2*) | -.233 | .660 | .436 | 1 |
| **ADD** (additive margins from the crossing) | -2.49 | 2.59 | 6.70 | **-65** |
| OR | +3.96 | 4.23 | 17.9 | -182 |
| L23 | +1.62 | 1.92 | 3.68 | -33 |
| L12 | +2.68 | 2.88 | 8.27 | -81 |

ADD over-predicts errors: first-order margins in each gap extrapolate across the gate kinks of the other line, the same
failure as T292's lin and T319's smoothing. It agrees with J1 on 95.4-95.8% of inputs, no better than P1 (95-96%).
On these 10 models the product's MSE share is small (.531 -> .436). On all 80 it is 40% one at a time.

**B2' (`python addB2p.py census | rms`, `addB2p_census.json`, `addB2p_rms.json`): J2's executed cost.**
Gate-pattern census on 100 x 100 over [0, .5]^2, per triple and half anchor, 10 models. These are lower bounds: the
200 x 200 grid on model 0 adds +5%.

| | cells per model, mean (min-max) | per triple per anchor | at 0.54 FPE per later cell + 2.0 per first | at the 2-D price 4.39 FPE per cell |
|---|---|---|---|---|
| nb anchors (V1 convention; b keeps its lines) | 35.4k (17.9k-61.3k) | 24.6 | **~21k FPE** | ~155k FPE |
| nb + b anchors (blind walked in 2-D) | 83.2k (36.4k-110.8k) | 28.9 | ~49k FPE | ~365k FPE |

J2's error vs truth (closure c remaining; budget.py's paired 2e5 inputs, 10 models): V1 convention bias +.214 / RMS
**.583**; no cut and blind walked +.124 / .470. Against the DRI's strict arithmetic (C <= ~3.45k all-in, <= ~1.2k extra beyond
C1's 3.3k), **J2's extra cells alone are >= 21k FPE even at the line-cell price, 17x over.** Strict is out of reach for
closure c via exact 2-D sets by arithmetic.

**B3 (`python corner.py`, `corner.json`): where the product's overstatement lives (mean-point closure c, not T339's
spread lows).**
- Product error D = P1 - J1, in V1's convention: net **+.574 pp** (sd .31), gross over 1.43 pp, under .86 pp.
- **Model dependence:** the net is **constant across models** (corr with 1 - p = .001, slope .00), consistent with
  T339's constant offset. The gross over mass scales with the error (corr .62, slope .077).

Share of the net (and over) mass by region; input mass in brackets:

| delta | corner g12, g23 < delta | L-band min(g12, g23) < delta | bias removed by the corner fix | corner fix extra FPE (0.54/cell, mean / max) |
|---|---|---|---|---|
| .025 | 2% (3%) [.2%] | 33% | .010 | 1,511 / 3,893 |
| .05 | 14% (13%) [.8%] | 66% | .081 | 2,575 / 4,407 |
| .1 | 49% (39%) [3%] | 91% | .283 | 4,047 / 6,382 |
| .2 | 92% (74%) [10.5%] | 101% | .528 | 5,599 / 8,724 |
| .5 | 100% | 100% | .574 | 7,910 / 13,694 |

**The corner hypothesis fails:**
- The overstatement lives in the **L-band along both axes** (one gap small, the other up to ~.2), not in the corner
  of both gaps small.
- The corner fix's cost is dominated by cell density near the ties. The cheapest delta that removes >= .25 pp (.1)
  costs ~4.0k FPE extra, even at the line price.
- **Go (>= .25 pp at <= +600 FPE): not met.** The half-anchor version (D2 = P2 - J2) is the same (.258 pp at delta .1).
- [inferred] A constant, model-independent ~+.5 pp offset concentrated near the two tie walls is the product's
  structural bias. Being model-independent, it could in principle be removed by a model-free correction, but that is
  a constant tuned on development models (inductive "advice") unless derived.

---

## T347: a first-principles account of the line-set product's overstatement. Pre-registered 2026-10-18 before any T347 prediction.
**Stage 0 (already measured, disclosed before the design).** On all 80 DEV2 models (budget.py's paired results),
the product's overstatement D_m = V1 - J1 (V1 convention) has mean +.568 pp but **sd .275**:
- range -.04 to 1.43;
- corr with 1 - p = .07;
- fused .571, torch .564.
**Even the best constant gives a mean |D_m - c| = .207 pp > .1.** So no model-free formula can meet the DRI's per-model
go line (mean |error| <= .1) on DEV2. A model-free derivation can at best remove the mean offset. A 5-model diagnostic
(development) shows the over-mass (1.0-1.6 pp) has the joint picking p1 or p3 about equally, with both gaps small
(medians .04-.10), against an under-mass of .55-1.2 pp. D is a difference of two ~1 pp model-specific terms.

**Derivation (first order; frozen in `t347/tilt.py` before DEV3 exists).** On each tie wall the joint correct set is
bounded by a curve in the other gap. The product replaces it with its value at the transverse point.
- Near the 2~3 wall: joint correct iff g23 > theta23(g12); the product uses theta23* = theta23(g12*).
- To first order theta23(g12) = theta23* + s23 (g12 - g12*), with tilt s23 = -beta/alpha, where:
  - alpha = d(binding margin)/d g23 along line 23 at its wall crossing;
  - beta = d(binding margin)/d g12 at the same point (one forward-mode tangent along line 12's direction, gates
    frozen: deductive).
- Then D23 = int_0^tmax dg12 phi(v2+g12) [Psi23(theta23(g12)) - Psi23(theta23*)], with theta clipped to [0, tmax] and
  Psi23(c) = Phi(v2)^8 - Phi(v2 - c)^8 (the exact model-free mass of g23 in [0, c]).
- Likewise D12 on the 1~2 wall, with tilt s12 from a cross tangent along line 23's direction at line 12's crossing,
  and Psi12(c) = Phi(v2 + c) - Phi(v2).
- D_pred = mean over triples of sum over the G = 20 v2 nodes (nb region only; in b the product is not used) of
  w 90 phi(v2) (D23 + D12). theta* = the start of the first correct interval of the walked line (from the candidate's
  own walk); a wall with theta* = 0 contributes 0.
- **Model-free content:** the order-statistic density and its conditional means. **Model-specific:** theta* and the tilts,
  read from the candidate's walks plus 2 extra tangents per triple (cost reported).
- **Check that the slopes cancel:** is D_pred nearly the same for all models (sd < .1), or does it follow the tilts?

**Test.**
- Predicted vs measured D_m on all 80 DEV2 (no tuning: D_pred has no free parameter), and on DEV3 when open. The
  formula and code are pushed before DEV3's truths exist. D_m on DEV3 = V1 - J1 by budget.py's pointwise form.
- **Go:** mean |D_pred - D_m| <= .1 pp on both. Also reported: corr(D_pred, D_m), and the mean offset error |mean D_pred - mean D_m|
  (a panel-level secondary).
Labels: D_pred is deductive given the walks (no sampling). D_m is diagnostic sampling. Cap 2 h.

### T347 results on DEV2 (`python t347/run_dev2.py`, `t347/dev2.json`; formula frozen at 790de13d before the test)
| | value |
|---|---|
| mean \|D_pred - D_m\| | **.230 pp**: **no go** (line .1; the best constant gives .207) |
| corr(D_pred, D_m) | .54 |
| mean D_pred / mean D_m | +.452 / +.568 (panel-level offset error .12) |
| sd D_pred / sd D_m | .273 / .275 |
| decomposition of D_pred (mean) | 2~3 wall + 1~2 wall; walls hit on ~49% of triple-lines; median tilts ~ +-.01 |

- **Slopes do not cancel.** The prediction varies across models as much as the truth does (sd .27), and it tracks it
  partly (corr .54). The overstatement is a property of each model's wall tilts, not of the order-statistic geometry
  alone. The geometry supplies the weights: the exact density and the mismatch between g12* and E[g12 | wall].
- **First order is not enough per model.** Most of the mass is near the corner (B3), where the tilt extrapolated from
  the crossing at g12* = .53 (or g23* = .35) down to gaps ~.05 is unreliable. That is the same failure mode as ADD.
- **Post hoc (not pre-registered), as an estimator correction:** V1 - D_pred moves the bias from +.439 to **-.013** and the
  RMS from .796 to **.667** on the 80 DEV2 models. The oracle product removal (J1) gives -.129 / .616. So the deductive
  correction recovers ~70% of the oracle's RMS gain.
  - Cost: 2 crossing tangents per triple. As counted in tilt.py, 2 x 720 x 3 FPE = 4.3k FPE. Reusing the walk's cell
    state at the crossing, one extra tangent is ~1 FPE (10 x 560 FLOPs + readout), so ~1.4k FPE.
  - This is a lever for the lenient line and for bias, not the pre-registered go.
- **DEV3:** the frozen formula is ready. `run_dev2.py` with DEV3's weights and budget.py's D_m (V1 - off_product) gives
  the out-of-sample test. DEV3 does not exist yet (T342). The DEV2 go has already failed, so DEV3 mainly tests the
  post-hoc correction out of sample.

**T347 amendment (DRI 13:44; written before DEV3 exists, after the DEV2 results above).**
- **Primary go:** the frozen formula (790de13d, unchanged) predicts DEV3's *panel mean* overstatement within +-.1 pp:
  |mean D_pred - mean D_m| <= .1 on DEV3, with no DEV2 tuning.
- Secondary, reported only: the per-model mean |error| and corr (the share of the sd explained).
- **Disclosure:** on DEV2 the same panel-level quantity is .116 (pred .452 vs meas .568), just outside +-.1. The formula
  is not changed in response.
- Caveat: D is measured and predicted for the mean-point closure c (T317d), not T339's spread closure. Its offset under
  spread lows may differ; that needs its own D_m.

---

# T351 (T317h): T317f + T339 spread lows + T347 tilt correction. Pre-registered 2026-10-18, before any T351 run.
**Sources (documented copies in `t351/`):**
- `cont_f.py` = claude/t218-tie-tail-labels @88453669 m16/approaches/r1_cont/cont_f.py, sha256 327dbb4a6ddc... (matches
  T339's record);
- `cont_g.py` = claude/t286-cheap-frontier @886e31ac m16/approaches/closure_floor/cont_g.py, sha256 d34dde3699c6...;
  verified: cont_g = cont_f + T339's `t339_diff.patch` exactly.
**Declared addition (`t351/cont_h.py`, the only new code).** `est_h(model, spread, tilt)` runs cont_g's est loop
unchanged (spread 0 = T317f, -1 = T339). If tilt is on, it adds T347's frozen tilt correction (t347/tilt.py, 790de13d),
with three declared implementation changes:
1. **Walks are not repeated.** theta* = the first correct start of each nb line, read from the estimator's own merged
   intervals (cont_g walk_c1 with screen C1, so screened segments count as correct, as in the estimate).
2. **Crossing slopes**, per nb line row with 0 < theta* < tmax: one value pass plus 2 tangents (along the line and along
   the other line's direction) at theta* + 1e-9, gates frozen.
   - Charged as executed: per step (2H^2 + 2H) + 2H + 2 x F_TAN, readout 3 x 2CH, margins/argmin 6C, tilt 4.
   - That is ~17.8k FLOPs (~3.0 FPE) per wall row.
3. **The g-rule is 8-point Gauss-Legendre** on [0, tmax] (40 in tilt.py), for cost.
   - Per wall row and v2 node: NQ x (one ndtr + power for Psi + clip/tilt arithmetic), plus one Psi at theta*.
   - Model-free weights (phi(v2 + g), the f23 density, Phi(v2)^8) are tables, uncharged.
   - D_pred(NQ = 8) vs tilt.py (NQ = 40) is reported on DEV2 as a check.
The estimate is est - mean over triples of sum_nodes w 90 phi(v2) (D23 + D12). The b region gets no correction.
**Arms:** A0 = T317f (spread 0, no tilt); A1 = spread -1; A2 = T317f + tilt; A3 = spread -1 + tilt.
**Checks before scoring:**
- A0 equals cont_f.est bitwise (estimates and FLOPs) on all 80;
- A1 equals T339's s = -1 row (RMS .884, bias +.758, C 3,748).
**Selection (one freeze):** min over the 4 arms of DEV2 MSE x C. MSE = mean_m (est - truth)^2 in pp^2 over 80 DEV2 with
the 1e8 truths; C = mean FPE as executed. F = mean e^2 x max(C, 1000) / mean p(1-p) is also reported, with bias, sd, RMS
and C for each arm. The selected arm is frozen as `t351/estimator_h.py` (est(model, budget=None) -> (estimate, FLOPs)).
**DEV3 check (pre-registered, no retuning):** the frozen arm and A0 (T317f) on DEV3, with its open truths. Report F for
both; "beats T317f" iff F_h < F_T317f on DEV3. The DRI then decides on HP5.
Labels: the estimator is sampling-free at runtime (deductive + declared approximations, as T317f); the tilt correction
adds no fitted constant. Cap 2.5 h.

## T351 results (`python t351/run_arms.py`, `t351/arms_dev2.json`, `t351/arms.log`; all 80 DEV2, 1e8 truths)
**Checks:**
- A0 = cont_f.est **bitwise** in estimates and FLOPs on all 80 (re-verified on raw values; the in-script flag was a
  false negative from the FPE round trip);
- A1 reproduces T339's s = -1 row (RMS .884, bias +.758, C 3,748);
- tilt NQ = 8 vs tilt.py NQ = 40: max |diff| .0025 pp.

| arm | bias | sd | RMS | C (FPE, executed) | MSE x C | F |
|---|---|---|---|---|---|---|
| **A0 T317f** | +.489 | .671 | .827 | **3,326** | **2,276** | **3.14** |
| A1 + spread (s = -1) | +.758 | .458 | .884 | 3,748 | 2,931 | 4.04 |
| A2 + tilt | **+.038** | .673 | .670 | 6,124 | 2,750 | 3.79 |
| A3 + spread + tilt | +.413 | **.451** | **.609** | 6,543 | 2,426 | 3.35 |

**Selection (pre-registered, min DEV2 MSE x C): A0 = T317f.** So T317h = T317f: nothing new is frozen, and the DEV3
"beats T317f" check is moot by construction (it would compare T317f with itself).
- The tilt correction does what T347 found: bias +.49 -> +.04, and with spread the best RMS of the family, .609.
- But as executed it costs +2.8k FPE: ~355 wall rows per line x (3.0 FPE crossing pass + 0.82 FPE g-rule).
  C nearly doubles, and MSE x C does not fall.
- The spread's extra +.27 pp bias (T339) is only partly removed by the tilt: A3 still has +.41.

**Not pre-registered; for the DRI to decide on (needs its own pre-registration).** The crossing pass is the cost
driver, and most of it is avoidable:
- the walk already holds the crossing cell's gate pattern and dL, so alpha is free;
- beta needs only one tangent with the cell's gates (~1 FPE).
At that count A3's C is ~5.1k, MSE x C ~1.9k (< A0's 2.28k) and F ~2.6. Strict (F <= 1 at ~5k) needs RMS <= ~.38, so
this still does not reach strict.

---

# T353: cost-only reimplementation of A3 (spread + tilt) reusing the walk's crossing-cell state. Pre-registered 2026-10-18 before any run.
**Change (`t351/cont_h2.py`).** The nb walks use a copy of cont_g.walk_c1 that also records, for each row, the state of
the cell where its first correct piece starts: gates G, dL, and L at theta*. This applies to screened pieces too.
- alpha = dL_k - dL_rival is taken from that state (free).
- beta = one tangent through the recorded gates along the other line's direction, plus its readout.
- The rival is the argmin of the margins at theta* (L affine from the cell).
- The g-rule, the integral and everything else are A3's code, unchanged.
**Charges as executed** (per wall row):
- L(theta*) affine: 2C;
- margins + argmin: 3C;
- alpha: 1;
- cross tangent: T x F_TAN + 2CH;
- beta and tilt: 5;
- the g-rule: 4,880 (as A3).
The recording itself is a copy, not arithmetic (uncharged, stated).
**Equality:** the target is estimates bitwise equal to A3 on all 80 DEV2. Declared now: if not bitwise (A3 recomputes a fresh
forward at theta* + 1e-9, while the reuse uses the walk's affine state, so floats may differ ~1e-15), the gate is
**max |est - A3| <= 1e-9** absolute. Both the max difference and the count of bitwise-equal models are reported. Stop if the gate fails.
**Go line:** DEV2 MSE x C <= .90 x T317f's 2,276 = 2,048 (A3's MSE is unchanged, so this is a cost test).
**If go: DEV3 check (pre-registered, no retuning).** T353 vs T317f on the same DEV3 models (the full 100 from the T192 child,
with open truths). Report F, MSE x C, bias, sd and RMS for both. "Confirms" iff MSE x C(T353) < MSE x C(T317f) on DEV3.
