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
