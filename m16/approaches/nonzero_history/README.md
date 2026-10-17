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

## Landmark rescue outcome
The separate landmark rule passes the zero-history reproduction control:
maximum discrepancy5.87e-10, below the omitted y-tail bound1.974e-9. Every
GL16/32 pair agrees within1.11e-16, and full geometry partition defects are
at most4.45e-16. These are numerical controls, not a floating-point proof.

| case | zero-history error | plugin-mean error | exact2-seeded error |
|---|---:|---:|---:|
| T4 base | -.00544372 | -.00546674 | -.00593840 |
| T4 relative .02 copy0 | +.03126919 | +.03160073 | +.03621509 |
| T5 base | -.00623715 | -.00969401 | -.00969678 |
| T5 seed2 | -.01127019 | -.02142358 | -.02254915 |

Both nonzero mean recipes worsen all four frozen pilot cases. In T4,
exact2 is the actual prefix mean, so its failure cannot be blamed on mean
closure error. Averaging the state loses information needed by the nonlinear
suffix; this does not identify all missing information or exclude richer
history approximations. The initial GL16/32 numerical failure and its
weight-derived landmark rescue remain separately recorded.

The twelve production estimates use960–2560 outer nodes and60–160 affine
panels each; CPU .0026–.0158 seconds in this short run. Full preparation and
both quadrature orders use1.16CPU seconds. Counts include affine construction,
line intersections, mean setup, special functions and per-interval powers;
these event counts are not a complete portable arithmetic/FLOP ledger.
In particular, scipy/NumPy kernels are not assigned fictitious unit costs.
No own-cost or matched-sampling winner is claimed from this diagnostic pilot.

Finite alternatives checklist:
- Tried zero, recursive deterministic mean, and exact-first-two-step mean
  followed by closure. Exact mean is the stronger frozen rescue forT4.
- Tried full affine topology plus Gaussian-width panel rescue, constant and
  tie controls, scaling, independent scalar mean integration, zero-history
  reproduction and order doubling. Numerical error no longer explains the
  checked percent-level discrepancies, subject to independent review.
- Not tried conditional first-input-sign/history mixtures, covariance-aware
  approximations, rank-conditioned state or longer exact history. These
  retain information discarded by every deterministic-mean variant and
  remain plausible; some overlap existing transport tasks. Do not infer a
  broad memory-model impossibility from this four-case pilot.
- No expanded28-case sweep: the frozen pilot's rescue criterion fails on all
  four cases. A different representation must be separately claimed/frozen.
- Independent direct-suffix validation requested from T55 before this scoped
  failure is treated as verified; root/DRI owns final classification.

Reproduction: use the standard one-thread capped review-lock command for
`controls.py`, `pilot.py`, and `pilot.py --normal-landmarks`. Reference loading
is confined to pilot.py; affine.py uses weights only and no input sampling.

Event-ledger correction: `np.where` evaluates both CDF-difference branches,
so cdf_mass performs four scalar CDF values per entry, not the two initially
counted. Corrected the counter and added explicit Legendre-rule construction
metadata. `accounted_landmark_pilot.json` reruns the unchanged numerical rule
with the corrected ledger; original pilot artifacts preserve the old counts.
All estimates are unchanged. The ledger remains event accounting, not a
complete special-function or linear-algebra scalar-FLOP conversion.
