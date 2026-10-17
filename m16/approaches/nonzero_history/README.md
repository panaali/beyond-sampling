# T76 nonzero history rescue

## Question
Does replacing the omitted hidden state by a weight-derived nonzero mean
repair the independently verified two-input suffix failures? Claim45dd8d7
was pushed before implementation. This folder is independent of the prior
h2_boundary_mechanism engine and keeps all its results intact.

## Method
Approximate the hidden state before the final two inputs by a deterministic
vector mu; resolve both remaining ReLU steps and readout boundaries exactly
as affine inequalities. Integrate final scalar x analytically, then integrate
first suffix scalar y by positive panel quadrature. The surrogate depends
only on its two suffix inputs, so the earlier m=T-2 iid ranks can be averaged
exactly despite the history approximation.

Freeze three variants: zero history (independent reproduction control);
plug-in mean recursion from zero, mu_next=E[relu(Wh*mu+Wi*X)]; and exact first
two-step mean followed by plug-in recursion for any further prefix steps.
No constants are fitted to outputs or saved accuracies. For the first two
steps let cplus=Wh*relu(Wi), cminus=Wh*relu(-Wi). Each mean coordinate is
(hypot(cplus,Wi)+cplus+hypot(cminus,Wi)+cminus)/(2*sqrt(2*pi)). For negative c,
use Wi^2/(hypot(c,Wi)-c) to avoid cancellation. This mean is exact for T4's
two-step prefix; replacing its distribution by its mean is still approximate.
T55 independently verified the formula and warned that mixed-sign Wh makes
the later plug-in recursion neither exact nor a general mean bound.

For fixed y, split every accepted final-input interval at x=y. The true
last-label probability has primitive Phi(x)^(m+1)/(m+1) below y and
Phi(x)^m-m*Phi(x)^(m+1)/(m+1) above y. The first suffix label has constant
weight Phi(y)^m above y and m*(1-Phi(y))*Phi(y)^(m-1) below. Each omitted
label receives (Gaussian interval mass minus the two suffix truth masses)/m.
T55 independently checked these formulas and the surrogate-only scope.

Construct a vertical affine arrangement: split y at first-step gate roots;
for every final gate/readout candidate, collect x lower/upper affine bounds,
all intersections of those lines and x=y, and y-only constraints. On every
remaining panel the active x interval is affine in y. Exact-zero readout
margins reject later labels by the first-index tie rule; zero gates must own
one branch. Normalize homogeneous inequality scales before sign decisions;
never use output truths to classify pieces.

Integrate y only over[-6,6], with missing accuracy at most2*Phi(-6). No input
truncation is imposed on x or omitted prefix inputs. Production GL16 per
panel; GL32 is a diagnostic refinement, never substituted after seeing truth.
Full floating geometry and quadrature error are not certified. Record all
construction, intersection, CDF, polynomial and node counts plus CPU.

## Status
Protocol frozen before measurements. First controls and pilot only.

## Frozen alternatives and falsifiers
1. Missing mean state explains the old failures: exact-two-step mean should
   improve the failed T4 relative-noise .02 copy0 beyond quadrature error.
2. Distribution/rank dependence matters more than the mean: the exact-mean
   surrogate can still fail; this is not evidence against every memory model.
3. Affine integration/mean closure is numerically wrong: zero-history must
   reproduce the existing independent two-input values; constants give1/T;
   Wh0 is history-invariant; positive coordinate scaling preserves estimates.
4. A better estimate costs too much: count all affine arrangement work;
   compare existing-reference errors against the old cost and own charged
   cost only after a separately reviewed operation convention. No FLOP win
   follows merely from fewer nodes or smaller CPU.

First pilot cases are fixed now: T4 base, failed T4 relative .02 copy0, T5
base and failed T5 seed2. Existing weights/reference files only; no new MC.
Run all three variants on every pilot at GL16 and32, with no selection.
If quadrature refinement is material, record failure before any expanded
ensemble. Strongest planned rescue is the exact first-two-step mean versus
naive plug-in mean, already included. A future conditional/random history
representation is distinct and untested; do not silently add it here.

## Results
Pending. Reproduction commands will be recorded with each bounded cycle.

## Compute accounting
One thread,2GB RSS,<=110CPU per cycle, shared compute lock; <=20CPU controls
may use the separate review lock. No model-dependent setup is cached across
estimates. Saved truths enter validation scripts only.

## Claims
[inferred] Rank averaging and scalar primitives are exact for the fixed
surrogate in real arithmetic. [speculative] Nonzero mean history improves
accuracy at useful cost. No challenge solution or certified error claim.

## Alternatives not ruled out
Conditional history, its variance/support, longer exact suffixes, and direct
boundary representations. Existing T91/T67 transport engines are not copied.

## Failure modes
Near-parallel affine boundaries, numerical cancellation of tiny interval
masses, positive-mass ties, zero input weights, and expensive arrangements.
All failure rows remain in artifacts; never silently clip negative masses.

## First controls and pilot: retained numerical limitation
The constant-class control caught an implementation defect before trained
runs: the below-y contribution was overwritten by the above-y contribution
rather than summed. `initial_control_failure.json` records .3151925 instead
of .25. After correction, independent one-dimensional mean integration
agrees within7.11e-15; zero/identical readouts, zero inputs, Wh0 and coordinate
scaling controls pass (`controls.json`,.036CPU).

The frozen four-case GL16/32 pilot is retained in `pilot.json`. Zero-history
reproduction errors reach2.39e-6 and order changes3.82e-5. Therefore the
initial affine-panel quadrature does not meet the frozen numerical control;
no ensemble or efficiency conclusion follows. T5 seed2 is already stable:
zero .29350526534 vs prior .29350526549, plugin .28335188003, exact2
.28222630452, all below the full reference .30477545878. The mean variants
worsen this checked example, but other cases need numerical rescue first.

Freeze a distinct numerical rescue before running it: topology changes alone
do not resolve steep Phi(a*y+b) transitions for large active-boundary slopes.
Split every retained vertical region additionally wherever its active lower
or upper affine bound equals a fixed integer in[-8,8], and at integer y
values. This targets Gaussian transition width using weights only. Use the
same GL16/32 and report all extra intersections/nodes, preserve the original
rule as default and its results. Require zero-history reproduction within
1e-8 plus explicit missing-y mass; no fitted breakpoint locations. If this
fails, the scoped pilot remains numerically inconclusive; do not begin a
full sweep or blindly increase order. Independent code review requested.
