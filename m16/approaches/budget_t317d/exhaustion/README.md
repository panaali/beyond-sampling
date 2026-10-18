# T356: strict-bar exhaustion checklist for the T317 paradigm (AGENTS rule 5)

↑ [budget_t317d](../README.md) · [TASKS](../../../../TASKS.md) · [INDEX](../../../../INDEX.md) section 4b

## Question
Can any variant of the paradigm "point closure of the low inputs + exact piecewise-linear gate walks along the two gap
lines per position triple + a declared combination rule" meet the **strict** own-cost bar (F <= 1) on retrains? This
document lists what was tried, what was not and why, and the quantitative floors, and gives a scoped verdict. A
different agent re-checks the key failure afterwards (rule 5b).

Scoring: F = mean e^2 x max(C, 1000 FPE) / mean p(1-p), with C counted as executed (Directive 8, amendment 2). Mean
p(1-p) is .0725 on DEV2 and .0689 on DEV3. Numbers are from the INDEX and READMEs as cited; all are **unverified unless
marked**. My own new arithmetic here (floors on DEV3) comes from `floors_dev3.py` and `floors_dev3.json`.

## Method
- Collect every variant of the paradigm from INDEX section 4b and the rows T300-T353.
- Restate each at its best panel score.
- Measure the two quantities that bound any variant (closure floor; integration error) on the fresh DEV3 panel.
- Compute the strict line at each cost.

## Status
Done (writing plus small arithmetic, ~5 CPU-min). **Verdict: strict negative (unverified), scoped**, below.

## 1. Variants tried (paradigm members; best panel numbers)
| task | variant | panel | bias pp | RMS pp | C FPE | F | outcome |
|---|---|---|---|---|---|---|---|
| T300 | R0 integral, frozen T297 cheap designs | DEV2 | +.21 | 1.57 | 1,752 | 5.97 | lenient by cancellation; ~90 without it |
| T302 | R0 exact gap-line walks (first cancellation-free) | DEV2 | +.42 | .75 | 26.3k | 20.6 | fail |
| T304 | R1W V1: closure c (v4 at latest free position, K = 1 node) | DEV2 | +.45 | .795 | 13.5k | 11.7 | narrow lenient fail |
| T304 | V2 (K = 2 v4 nodes) | DEV2 | | | | 20.1 | worse |
| T306 | two transverse points per region (B) | DEV2 | +.78 (from +.45) | | | | worse (explained by T324 A) |
| T307 | + B2 prefix sharing + exact x9 marginalisation | DEV2 | | floor .503 -> .461 | | | stop (< 10% floor cut) |
| T311 | adaptive skipping of no-flip lines | arithmetic | | | best ~7.8k | | killed by arithmetic |
| T315 | capped walks (first c cells, then extrapolate) | DEV2 | -10.4 to -5.3 | | 4.3-9.8k | 900-1,700 | fail |
| T317/b/c | cost-only rewrites (continuity restarts, B2, G = 20, merges) | HP2 / HP3 | +.42 to +.53 | .81-.99 | 5.4-7.9k | HP2 6.8-9.7; HP3 5.08 | lenient pass |
| T317d/e | convention clean-ups (no MC; exact points) | HP3 | +.41 | .80 | 5.4k | 5.07 | lenient |
| **T317f** | + C1 frozen-gate margin screen | **HP4 (sealed)** | +.436 | .749 | 3,302 | **2.63 [1.93, 3.39]** | **best; lenient pass, strict fail** |
| | | DEV3 | +.425 | .721 | 3,307 | 2.50 | |
| T318 | kinked-tangent surrogate (first-order event expansion) | DEV2 (10) | | 6.0-13.3 vs closure | | | stop |
| T319 / T327 | Gaussian-smoothed verdict; cone-clipped J; 1-D exact largest low | DEV2 | | .44-1.33 floors | +6-18k | | dead (verified, scoped) |
| T326 | T317e knob frontier (tmax x cell cap x lines), 80 configs | DEV2 | | | floor ~2.2k | none beats T317e | negative, scoped |
| T330 / T330b | rigorous safe-to-tmax screens | DEV2 | exact | | 5,525-5,561 | | saves nothing |
| T331 | tmax 1 / 2, affine extension, + C1 | DEV2 | +.10 | .669 | +53% | MSE x C 1.08x | no |
| T333 J1 / J2 | exact joint 2-D set at 1 / 2 anchors | DEV2 (10) | -.23 / +.21 | .66 / .58 | J2 >= ~21k extra | | out of reach by arithmetic |
| T333 B1 | zero-cost combination rules (OR, single line, AVG, additive margins) | DEV2 (10) | +1.6 to +4.0, ADD -2.5 | 1.9-4.2 | 0 extra | | all far worse |
| T333 B3 | corner-only exact set | DEV2 (10) | removes .08 / .28 pp | | +2.6k / +4.0k | | no go |
| T338 / T341 | mechanism-informed and order-statistic closures | DEV2 (oracle) | | floors .31-.60 | | | no go |
| T339 | spread lows s = -1 | DEV2 / DEV3 | +.758 / +.645 | .884 / .836 | 3,748 | 4.04 / 3.77 | no freeze; sd gain does not replicate (T343) |
| T347 | tilt correction of the product (frozen formula) | DEV2 / DEV3 | per-model MAE .23 | | ~1.3k | | per-model no go; panel mean within .063 on DEV3 |
| T351 | 4 arms: T317f / spread / tilt / both | DEV2 | +.04 to +.76 | .61-.88 | 3.3-6.5k | 3.14-4.04 | selection keeps T317f |
| T353 | spread + tilt with walk-state reuse | DEV2 / DEV3 | +.413 / +.340 | .609 / .673 | 5.08k / 5.06k | 2.60 / 3.32 | DEV3 does not confirm |
| T353 post hoc | tilt only with reuse (A2-cheap) | DEV2 / DEV3 | +.04 / **-.015** | .670 / **.606** | 4.66k / 4.64k | 2.89 / 2.47 | ties T317f on DEV3 |

Diagnostics bearing on the paradigm:
- **T324 / T334** error budget (the product and tmax terms are confirmed): the walk approximations carry 66% of the MSE,
  closure c 34%. The product is -.57 pp bias and -40% MSE one at a time.
- **T308 / T314 / T332** information ceilings: cheap per-class features hold out to .48-.86 pp. T332, the paradigm's own
  per-row outputs, gives .577: neither go nor stop.
- **T305:** position sharing gives no >= 4x class cut.

## 2. Variants not tried, and why
| variant | why not (or status) | expected effect |
|---|---|---|
| second cross-gap anchor for the first wrong piece (Fable T347 note, C3) | proposed after T351; not funded yet | ~.25-.3 pp of bias at +0.4-1.1k FPE. Bias only: the variance floor (section 3) remains |
| second-order tilt (a'') at the existing crossing | not built; Fable M2 says it is a minor term | small bias, no variance gain |
| per-gate-pattern operator-norm screens; zonotopes (T330's untried list) | not built | estimate-preserving cost cuts of at most what T326's frontier allows (floor ~2.2k) |
| EM or truncated-Gaussian mixture states (T309 untried) | other paradigm (law propagation) | outside scope |
| v4 (or v5) at its true position | cost: ~4e5 FPE (T304 Stage A), v5 not cheaply placeable (T341) | floor .39 / .31 but unaffordable |
| fitted bias offset or regression on features | inductive (Hilton; T308/T314 are diagnostics only) | excluded by the rules |
| exact 2-D integration of the closure (J2) at full resolution | costed, not built (T333 B2') | >= 21k FPE extra (line price; ~155k at the 2-D price) |
| a different combination of the two line verdicts | covered by T333 B1 (6 rules incl. a geometry-derived one) | all far worse |
| more transverse points with the product kept | T306 B and T324 A: refining the anchor alone exposes the product's bias | worse |
| the HP5 sealed panel for any of these | not used (reserve; T353 did not confirm) | — |

## 3. Quantitative floors
**(a) The closure itself is not the binding floor at today's cost.** Closure c with its true v4 (T304's floor), integrated
perfectly, by paired MC (`floors_dev3.py`; 100 DEV3 models, 2e5 inputs each, DEV3 1e8 truths):

| | bias | sd | RMS | F if it cost 2.2k / 3.3k / 5k FPE |
|---|---|---|---|---|
| E0 (closure c, true v4) | -.129 | .414 | **.432** | .59 / **.89** / 1.35 |
| E1 (closure c, K = 1 node v4, T317f's closure) | +.173 | .431 | .462 | .68 / 1.02 / 1.55 |
| DEV2 (T324 budget) E0 / E1 | -.154 / +.155 | | .463 / .475 | |

**(b) The binding floor is the walk-level integration error.** On DEV3, T317f - E0 per model has mean +.554, sd .413 and
RMS **.690**. It is uncorrelated with the closure's own error (corr .00), so the errors add in quadrature: sqrt(.413^2 +
.414^2) = .585, which matches T317f's sd of .586. Strict at cost C needs total RMS <= 100 sqrt(mean p(1-p) / C):

| C (FPE) | strict RMS line (DEV3) | integration error allowed beyond E0 (quadrature) | today (T317f) |
|---|---|---|---|
| 2,200 (T326's cost floor) | .560 | .357 | .690 |
| 3,307 (T317f) | .457 | **.149** | .690 |
| 5,000 | .371 | 0 (E0 alone already exceeds the line) | .690 |

So strict needs the integration error cut **4.6x at no added cost** (or ~2x while also cutting C to the 2.2k floor).
- Removing only its mean (+.554; the tilt removes ~+.44 of it, T353 post hoc) still leaves its sd .413 > .149.
- **Variance-only F** (bias set to 0, each arm's own sd and C):

  | | T317f | s = -1 (T343) | T353 | A2-cheap |
  |---|---|---|---|---|
  | DEV3 | **1.65** | 1.54 | 2.50 | 2.50 |
  | DEV2 | 2.07 | 1.08 | 1.42 | 2.91 |

  On fresh models no arm reaches F <= 1 even with a perfect bias fix. The DEV2 s = -1 figure (1.08) did not replicate.

**(c) Costs.**
- T326: the family's cost floor is ~2.2k FPE (strict F ~1.9 at unchanged RMS).
- T333: J2 (the exact joint set at 2 anchors) is >= ~21k FPE extra at the line-cell price; ~155k at an honest 2-D price.
  The corner-only version costs >= 2.6k for .08 pp.
- T344 / T311 / T330: certificates are infeasible (interval bounds 1e8-1e9 per unit length; boxes certified only <= 1e-4
  wide). Rigorous screens save nothing.

## 4. Scoped verdict
**Strict: negative (unverified), scoped to point closure + exact gap-line walks + declared combination rule (the T317
paradigm: R0, R1W, T317-T317f and its tilt/spread variants), on zoo-config retrains, at C <= ~6k FPE.**
- The best member scores F 2.63 [1.93, 3.39] on the sealed HP4 and 2.50 on DEV3.
- With a perfect deductive bias fix the variance alone gives F >= 1.5 on DEV3 at ~3.3k FPE.
- The only measured route to the needed integration accuracy (exact 2-D sets) costs >= 6x the allowance.
- This is not a proof. It rests on:
  - the integration schemes tried (1-D walks, line products, zero-cost rules, first-order corrections);
  - the closures tried;
  - the panels DEV2 / DEV3 / HP2-HP4.
- **Not ruled out:** an integration scheme of a different kind that reaches <= ~.15 pp integration RMS against the
  closure at <= ~3.3k FPE, or <= ~.36 pp at <= 2.2k.
- **Not covered by this verdict:** other paradigms (law propagation, exact enumeration, rare events) have their own
  checklists.

## 5. The cheapest test that would overturn it
**An integration-error test against the closure's own oracle, before any estimator is built.**
- Pick a candidate integration scheme for closure c, for example Fable's second cross-gap anchor plus the tilt.
- On 10 DEV3 models, compute its estimate minus E0 (the paired MC of closure c, `floors_dev3.py`, ~1 CPU-min) and its
  executed C.
- **Overturn** if RMS(estimate - E0) <= 100 sqrt(.0689 / C) - [the E0 term in quadrature], i.e. <= .15 pp at 3.3k or
  <= .36 pp at 2.2k, with a mean within +-.1.
- Expected cost: < 30 CPU-min per candidate.
- No variant tried so far is within 2x of this line: the best integration RMS against the closure is T317f's .690 on
  DEV3. With the tilt it is about .55 (est.), because the tilt removes the mean but not the sd .413.

## Compute accounting
`floors_dev3.py`: 100 models x 2e5 x 2 closure verdicts (~3 CPU-min). Everything else is arithmetic on existing results.

## Claims
1. [measured] On DEV3, closure c with perfect integration has RMS .432 (true v4) / .462 (node v4). At T317f's cost that is
   F .89 / 1.02, so the closure is not the binding strict obstacle at ~3.3k FPE.
2. [measured] T317f's integration error against E0 is .690 RMS (sd .413), uncorrelated with the closure error.
3. [inferred] Strict requires the integration error <= ~.15 pp at ~3.3k FPE. No tried scheme is within 2x, and the only
   accurate one (J2) costs >= 21k extra.
4. [measured] With bias removed, variance-only F on DEV3 is >= 1.54 for every arm.

## Alternatives not ruled out
- A non-walk integration of the closure (for example adaptive 2-D quadrature with shared cells across triples, or
  symbolic integration over gate-pattern regions) could have a different cost-accuracy curve. It is untested.
- The DEV2 -> DEV3 shift (the spread's variance cut vanished) shows that development-panel gains can be panel-specific.
  Variance-only F could also be lower on some other panel.

## Failure modes
- The verdict depends on the cost convention (Directive 8). Under a convention that charges less for walk cells, the
  allowance grows (e.g. at 2.2k it is .357 pp).
