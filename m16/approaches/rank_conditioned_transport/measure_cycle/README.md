# T67: target-measure conditioning discriminator

## Question
Does the failed penultimate-cache GL2/4 stability gate reflect inverse-normal
conditioning of the outer target coordinate rather than spatial accuracy?

## Method
Retain the same h9 spatial grids, complete union of baseline and penultimate
physical target knots, and unchanged scalar kernels. Compare the existing
q-space rule with a physical u-space rule on exactly those panels. The latter
uses q=(Phi(u)-Phi(-6))/Z and density
T(T-1) q^(T-2)(1-q) phi(u)/Z, Z=1-2Phi(-6).
Both use target truncation [-4,4], full input support [-6,6], and retain raw
weights without renormalization. Tiny-panel omission bounds remain explicit.
Use local adapters only; previous code and artifacts are preserved.

## Frozen alternatives and falsifiers
1. Inverse-normal derivatives cause the outer discrepancy: physical-u GL2/4
   becomes stable below1e-7, whereas q GL2/4 retains the recorded discrepancy.
2. Finite panel smoothness or geometry causes it: both fail; do not call the
   representation impossible or simply launch higher-order/mesh searches.
3. A changed target measure creates a false benefit: compare each rule mass
   with exact Beta CDF endpoints; constant/duplicate outputs and the inherited
   first-index controls must pass before a trained run.
4. Coordinate improvement hides spatial error: report total and all label
   contributions against the already frozen independent cone references.
   A stable value near .30421 is still inaccurate; no component-selected mix.

First validate polynomial-in-q and positive Gaussian target-weight controls
on fixed artificial panels and the complete trained topology. Then h3
constant/duplicate full recurrences (raw omission-adjusted mass). Trained
pilot exactly four h9 cached estimates: q/u x GL2/4, identical union panels.
No baseline recomputation, new weights/noise, model parameter selection,
quadrature order beyond4 or mesh sweep. All initial and failed results stay.
Any further step needs a distinct question and a new claim.

## Status
Claim87ba500 pushed before implementation; protocol frozen now.

## Results
Pending. Previous q-space discrepancy1.77e-7 and total error-.00351888
are hypotheses to reproduce, not supplied to the estimator.

## Compute accounting
One thread,2GB, review lock<=20CPU for controls, primary lock<=110CPU/pilot.
Charge topology, GL construction, normal CDF/inverse-CDF, densities, Gaussian
moments/fallback, sparse arrays, recurrence and compilation if used. This
phase keeps Python kernels; no native compilation. Named event counts are
partial/unaudited; full CPU is measured, no portable-FLOP or sampling claim.

## Claims
[inferred] The transformed measure is identical in real arithmetic.
[speculative] Physical-target quadrature improves outer numerical stability.

## Alternatives not ruled out
Incomplete floating topology, near-coincident branches, piecewise analytic
integrand conditioning, earlier spatial interpolation, exact two-step terminal
representation, and rigorously bounded integration remain open.

## Failure modes
Missing Jacobian, raw target mass drift, first-index ties, inconsistent panel
omissions, inverse-normal rounding, and treating GL agreement as certification.
Independent review and Claude DRI classification are required.

## Control checkpoint
`controls.py` completed in1.462CPU. Artificial64-panel and complete1532-panel
rules were checked against exact Beta integrals of1,q,q^2; all GL4 errors
are below2e-9. Constant and duplicate-readout recurrences at h3 match their
raw omission-adjusted mass/T within1.67e-16 in both coordinates. T55
independently derives the Jacobian, Beta CDF Tq^(T-1)-(T-1)q^T, per-label
1/T factor and separate full-input omission. Full trained pilot follows.

## Frozen trained result: coordinate hypothesis not supported
All four integrations complete in50.983CPU seconds. q-space estimates
exactly reproduce the previous cached GL2/4 values to displayed precision.

| Coordinate | GL2 estimate | GL4 estimate | Absolute change | GL4 CPU |
| --- | ---: | ---: | ---: | ---: |
| q | .304211687608592 | .304211864775995 | 1.77167e-7 | 17.098 |
| u | .304211688729408 | .304211864742884 | 1.76013e-7 | 16.954 |

Both fail the frozen1e-7 refinement gate. GL4 coordinate disagreement is
3.31e-11 and raw target mass defects are below4.48e-14. Thus the tested
coordinate change does not remove the refinement discrepancy; a missing
Jacobian or target-mass normalization cannot explain its scale. This is
finite evidence against this particular conditioning explanation, not a
proof of convergence or a universal negative about target coordinates.

Both retain total bias-.003518878 and component errors approximately
[-.003060092,-.000002521,-.000456265,0]. Spatial approximation remains the
much larger unresolved error. No noise/grid/order expansion follows.
The stored stats include all cache and operator work, target nodes and
omission bounds. Full CPU measured, event counts partial and unaudited.

## Exhaustion checklist and handoff
Formulation: independent Beta/Jacobian derivation and omission-adjusted tie
controls pass. Representation: unchanged h9 and complete1532-panel union;
not a test of alternative spatial representations. Numerical alternatives:
q/u GL2/4 tested, neither passes strict refinement; higher orders, adaptive
local error localization and directed bounds remain untested because this
bounded phase tests coordinate choice rather than launching a sweep. Cost:
full cold setup/recurrence measured, no portable-cost or sampling assertion.
Independent source/results review requested. Only Claude may classify the
scoped negative; no family is declared dead. A concrete later discriminator
is localizing the signed GL2/4 difference by complete physical panel before
choosing a numerical repair, rather than increasing all rule orders.
