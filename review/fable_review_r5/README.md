# Fable review, round 5 (2026-10-18): where T317e's MSE x C goes, and whether strict is reachable

Read (nothing run; arithmetic only): SO §9 + §7-8; INDEX T286-T325 and H-L1; TASKS T324/T326/T327; my round-4 review;
review/candidate_redteam; r1_cont README through T326 (claude/t218-tie-tail-labels 2cff21bd, incl. the cost table, the
80-config frontier, the oracle floors and the "2.2k floor" note); r1_walk (T304/T306), r0_walk (T302 tmax scan),
kink_surrogate side finding (T318 K3). Child results are unverified unless INDEX says otherwise. No Codex/Astra r3.

Conventions. F = MSE x max(C, 1k) / mean p(1-p). On HP3 the constant is 694 pp²·FPE (F 5.07 at RMS .805, C 5,431);
DEV2 uses 740. Strict: MSE_pp² x C <= 694. T317e today: .648 x 5,431 = 3,519, so the gap is 5.07x.

## 0. The ledger (what multiplies what)

MSE .648 pp² = bias² .172 + variance .476. The variance splits into the closure-c floor (T304: RMS .471, mean -.156 ->
var .198), the declared approximations (T318 K3: exact walk vs closure-c MC, RMS .72, mean +.55 -> var .216) and a
residual covariance .062. Shares of MSE x C [inferred from measured parts]:

| term | pp²·FPE | share |
|---|---|---|
| approximation variance (one transverse point, product set, blind g12, tmax cut, mu-linearised lows, K=1 v4 node) | 1,173 | 33% |
| closure-c variance | 1,073 | 30% |
| bias² (+.415 pp: approximations +.55, closure -.16) | 935 | 27% |
| covariance | 339 | 10% |

Cost (T326, measured): prefix trees 1,729 + first-cell readout 317 = 2.05k FPE is pure label combinatorics (2,815
nodes per line, identical for every model); later cells 3.3k, of which 2.6k come after the row's last wrong piece.
Oracle exact-estimate floor 2.2k FPE. **Correction to T326's text:** at RMS .79 and 2.2k the strict F is 1.9, not 6.3
(the DRI has since made the same correction). That matters below.

Two facts drive everything: (i) every approximation removed so far cost at least as much in C as it saved in MSE
(V2: C x2, MSE x.86; V3: x4, x.80; T306-A: x1.18, x.975; T306-B: x2.1, MSE x1.53; T302 tmax .5->1: C x1.27, MSE x.80),
so the paradigm trades about 1:1 [measured]; (ii) the only moves that helped were exact cost cuts (B2, continuity,
merging) and those are now at the combinatorial floor [measured].

## 1. Designs, ranked. None reaches 5x on the evidence; I say so and give the multipliers

**D1. tmax upward, by exact walk and by free affine extension (bias + variance; exact in direction).**
*Arithmetic.* T302 on all 80 DEV2 (R0 walk): tmax .5 -> 1.0 moves bias +.42 -> +.28 and RMS .75 -> .67 (MSE x.80) at
C x1.27 before continuity; on 10 models the mean goes +.12 -> -.05 by tmax 6, so the cut is worth ~+.15 pp of bias
and ~.3 pp of RMS. Beyond .5 the cells are coarse (H-E15: 37-44x coarser far from ties), so in the continuity
implementation the extra cells should cost well under +27%: I expect +5-12% C. Projection: RMS .80 -> ~.69, bias
+.41 -> ~+.25, F 5.07 -> ~3.7-4.0 (x1.3). The zero-cost variant extends the last walked cell's affine margin to its
root beyond tmax and captures the "continuation" part of the beyond-.5 error.
*Relies on / contradicts.* Relies on T302's measured scan. Not killed by T315 or T326: those capped the walk *near the
tie* (c = 1-4 cells, or tmax .1-.3 inside the fine region), where frozen extension gives 2-14 pp; T326's knob grid
only scanned tmax *downward*. The affine extension beyond .5 is the untested coarse regime.
*Kill test (pre-registerable, ~5 CPU-min).* T317e with tmax in {1, 2} and with the affine extension at tmax .5, all 80
DEV2, charged as executed. Go: DEV2 MSE x C <= .80 x T317e's (F <= 3.75). Stop: C grows > 15% or the bias moves < .08.
*Failure mode.* Errors beyond .5 are islands (margin positive at .5, negative later), so the extension finds nothing
and the exact walk pays many cells for them.

**D2. T330: an estimate-preserving (or declared) safe-to-tmax screen, with an oracle pre-check from existing logs.**
*Arithmetic.* The oracle says 92-95% of non-blind later cells follow the last wrong piece: a perfect screen saves
2.6k -> C 2.2-2.5k, F 1.9 at today's RMS, ~1.6 with D1. That is the paradigm's ceiling on the cost side.
Realistically the saving per non-blind line is ~1.6 FPE (b-g23 .34), so the screen must cost <= .5 FPE per line and
fire on >= 70% of the post-last-wrong-piece cells; T311's certificates (Lipschitz 1e8-1e9 per unit length; CROWN 2-3
FPE per line) do not qualify, and a per-cell adjoint (~1 FPE) is cost-neutral at best.
*Relies on / contradicts.* Relies on T326's oracle floors. T311 killed *certificates*; it does not cover a declared
heuristic screen ("current cell's affine margin positive over the whole remaining window, after the first wrong piece
ends"), whose error is measurable. T326's cap-4 result (2.31 pp) shows a *count* cap fires too early; a *margin*
screen fires on a different, later set of cells.
*Kill test (from logs, no walks; ~minutes).* For every row, find the first cell whose affine margin is positive on
[t, tmax] and count the error mass in wrong pieces after it. Go: that mass has RMS <= .10 pp across the 80 DEV2 and
the screen fires before >= 70% of the post-last-wrong-piece cells. Stop: >= .25 pp (T326's cap pathology in a new
guise). Only then build it; a sealed HP4 is the claim.
*Failure mode.* The same mechanism as T315/T326: a gate flip after the screen fires turns the margin negative again
(T318: flips shift later slopes by O(1)), and the bias goes negative.

**D3. The approximation term, after T324, with the T306-B anomaly as the hinge.** Not a design yet; it is the
largest term and the only one whose fix *could* be nearly free.
*Arithmetic.* By differences (V1->V2 .16 pp bias and ~.30 RMS for the K=1 node; V2->V3 .07 and ~.19 for blind g12;
T302 ~.15/.3 for tmax), the residue of the +.55/.72 approximation error is ~+.2 pp bias and ~.5 pp RMS from the one
transverse point + product set + mu-linearisation. If that residue is a *combination-rule* error (how S23 and S12 are
combined), it can be fixed at zero cost; if it is a missing dimension (the 2-D (g12, g23) structure), every fix costs
lines (x1.5-2) and the 1:1 trade kills it. The evidence points the wrong way for us: T306-B (two transverse points)
*raised* the bias +.45 -> +.78, which a convergent approximation cannot do.
*Kill test (T324 should include it; 10 DEV2 models, closure-c MC binned in (v2, g12, g23) per triple, ~30 CPU-min).*
Compare the true 2-D correct mass per triple with (i) the product at the one point, (ii) the product at each half's
point (T306-B), (iii) min/union rules, (iv) the exact 2-D set. Go: a zero-cost rule cuts the residue's RMS by >= 40%
(then rebuild once, HP4). Stop: the exact 2-D set is needed (then D3 is dead by the 1:1 trade).
*Failure mode.* The error is genuinely 2-D and diffuse (ANAT-80's soft tail is 60% of the between-model variance).

Stacked best case (D1 + oracle-level D2 + a free D3 fix): F ~1.2-1.7. That is the honest ceiling of this paradigm.

Considered and killed by arithmetic, so T326's checklist can list them: (a) a single continuous path through all 720
tie points (continuity replaces the prefix tree): each transition between tie points crosses ~15-30 cells (8-16 FPE)
against .8 FPE per shared first cell; (b) a second tangent in the v4 direction to replace K=2: +~2.6k FPE for MSE
x.68, neutral; (c) swap-pair value sharing (the tie points of (p1,p2,p3) and (p1,p3,p2) are the same input) and
parent-matvec sharing in the tree: exact, ~7% + ~8% of C, worth doing but not a lever; (d) low-rank W_hh shadow:
T290's mechanism (small recurrent weights matter) applies, and rank < 8 is needed for any saving.

## 2. The lanes

**T326 (done, negative).** Right conclusion, two gaps. (1) It scanned tmax only downward; the one direction with a
measured favourable MSE trade (T302) is missing: D1 is a one-knob rerun. (2) Its oracle-floor reading was off by 3x
(F 1.9, not 6.3 at 2.2k); read correctly, the screen (T330) is worth more than the lane's text says, but it still
leaves F >= 1.6 even with D1, so strict additionally needs ~40% of the remaining variance removed.

**T327 (closure floor).** The go line (oracle floor <= .30 at <= +1.5k FPE), *if met*, gives DEV2 MSE .648 - .198 +
.09 = .54 at 6.9k: F 5.4, no better than today, because the approximation variance and the bias are untouched and C
grows. The lane only pays once the approximation term is fixed, or with a closure that is *free*. Two changes:
(a) state the go line as MSE x C (DEV2 MSE x C <= .85 x T317e's) so that cost growth is counted; (b) replace the two
first-order variants (cone-clipped J, 1-D largest-|J| low: the regime is gate-switching, T319 4-12% in cone) with
the one measured closure lever, spread lows (T298: R0 .515 -> .453; top-4 .389 -> .244), in the only form that is
free: a *position-deterministic* spread (low at position j at mu + (-1)^j sigma, one pattern). T326 shows the lows add
no tree splits because they share one value; a fixed per-position pattern is also identical across triples, so the
tree stays at 2,815 nodes and C is unchanged. Oracle floor for closure c + 1-pattern spread on 80 DEV2 (T298 protocol,
~15 CPU-min). Go: floor <= .40 (MSE -.07, F x.89 for free). Two patterns double C and cannot pay. Keep the T319
re-check (rule 5), it is cheap.

**Is "T330 + bias fix" the best strict route?** Yes, among concrete routes: it has a measured ceiling (F ~1.6-1.9)
and cheap pre-checks. No new geometry survives T326's requirement (reduce *distinct inputs*): sharing across classes
costs >= .31-.69 pp (T305) or 1.4-1.9 pp (T300); a tie-point path costs 10-20x the prefix tree; a 1k-FPE
first-cell-only estimator has a fitted ceiling of .855 pp (T314, inductive) and deductive realisations at 1.6-3.1 pp
(T300, T326). But T330 + D1 reaches ~1.6, not 1.0; strict needs D3 to be free as well. My odds for T330 reaching
<= 2.5k with estimates preserved or a declared error <= .1 pp: 20%.

The missing lane is the approximation term (33% of MSE x C). T324 is its diagnostic; D3's kill test should be added
to it, and T306-B must be explained before any new variant.

## 3. Structural assessment: is strict unreachable at 1-5k FPE?

[measured] (a) Cost floor of the 3-line geometry: 2.05k FPE of label combinatorics plus >= 1 later cell per row;
oracle exact-estimate floor 2.2k (T326). (b) Closure floors: c .471, R0 .515; top-4 .389 at x7 cost; spread .244 at
x14; undoing the x7 by class sharing costs >= .31 pp (T298, T304, T305). (c) Exact integration of a point closure by
gap-line walks has error .28 pp at 26k FPE with 4 lines (T302) and .72 with V1's cuts (T318). (d) Fitted ceilings:
.855 pp from <= 4k-FPE anchor features; .477 only with 25.8k-FPE rank_exact (T314, T308).

[inferred] The paradigm (point closure + exact 1-D walks along gap directions + per-triple enumeration) cannot reach
strict except in its best case with everything perfect at once: floor² x oracle cost = .22 x 2.2k = 486 (F .7); add
the best measured integration error (.28) and it is 660 (F .95); any realised design adds bias and later cells and
sits at >= 1.6. Along the closure ladder floor² x cost is .22-.27 (R0, c), 1.06 (top-4), .83 (spread): the product
never falls, so a lower floor cannot be bought. This is a "no free lunch on every measured axis", not an
impossibility: no bound covers deductive estimators outside this measurement class.

[speculative] The deeper reason is where the between-model variance sits: diffuse over most prefix nodes (T326: pruning
below 1e-4 absolute drops .72-.86 pp per line), over ~44 triples per weak model (ANAT-80), and in the lows' 6-7
dimensions where gates switch (T319). A deductive estimator at C FPE visits <= ~C/.54 cells; the variance it does
not visit must be extrapolated, and every extrapolation across a gate flip has been measured at >= 1 pp.

**Cheapest settling experiment (paradigm-level information ceiling, from logged outputs, CPU-minutes).** Fit, on
fused_s0-39 and test on torch_s0-39 (T314's protocol), a ridge predictor whose offset is the deductive estimate and
whose ~30 features re-weight T317e's own per-row outputs (deficits, tie margins, slopes, cell counts, aggregated by
line type and the ages of p2/p3). Go: held-out RMS <= .50 pp (F .79 at the 2.2k oracle cost: information exists;
look for the deductive mechanism behind the re-weighting). Stop: >= .62 (F 1.22 at 2.2k: no function of what this
geometry measures reaches strict even at its oracle cost; with T326, strict is closed for the paradigm). A fitted
offset alone gives .69 (F 3.7 at 5.4k), so the test is informative either way.

## 4. Bias: is there a deductive correction?

Partly. The +.41 pp is a sum of signed terms [inferred from V1/V2/V3 and T302 differences]: tmax cut +.15 (the gap
beyond .5 assumed correct), K=1 v4 node +.16 (error convex in v4 near the 3~4 side), blind g12 +.07 (assumed
correct), one point + product + mu-linearisation ~+.2, closure c -.16. The "assumed correct" terms have a deductive
sign (they can only overestimate accuracy) but no deductive magnitude short of walking them. The only free one is
tmax (D1: affine extension, or the exact walk into the coarse region); it removes ~+.15 of bias and ~.3 pp of RMS.
The K=1 and blind terms cost 1:1 (V2, T306-A). The closure's -.16 is not a bias of the integral and no correction
exists without a new closure. A deductive bias correction therefore caps at ~.25 pp residual bias (bias² .06 instead
of .17): F x.83 together with the variance it removes, ~x.73 in all. A panel-fitted offset is not allowed and would
give F 3.7.

Side note for the dossier: the DEV2 error slope is -.067 pp per pp of accuracy and the p > .95 subset has bias +.01
(red-team item 5); the bias is a property of weak retrains' soft tails (T296: +.09 pp per pp of tail error), which
is consistent with the "assumed correct" terms scaling with the model's error mass.

## 5. Odds (subjective)

Strict (F <= 1, upper CI too) on a fresh sealed panel within 3 months, deductive: 5% (was 7%). D1 alone improving
F to <= 4.0 on DEV2: 65%. T330 reaching <= 2.5k with <= .1 pp declared error: 20%. The paradigm information test
stopping (>= .62): 70%. If it stops and D3's test shows the residue is 2-D, I would close the strict question for
this paradigm as "dead (verified, scoped)" and report H-L1 with D1's improvement as the final lenient number.
