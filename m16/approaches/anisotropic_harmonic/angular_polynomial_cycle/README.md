# T56: matched nonperiodic angular basis

## Question
Does the periodic Fourier representation introduce avoidable cut-projection
error when the physical root domain is a finite lune interval?

## Method
Keep exact root support, radial degree8, radial Q64 sine-mapped panels and
81 real degrees of freedom. Replace angular Fourier modes-4..4 by normalized
Legendre polynomials sqrt(2m+1)Pm(t), m0..8, where
t=2(phi-phi0)/L-1 and conditional angular measure isdt/2. Root constant and
whole-sphere mass1/12 stay exact. On each clipped angular interval use a
positive9-point Gauss-Legendre rule: angular products have degree<=16, so
this integrates the Gram entries exactly in real arithmetic. Use the same
real matrix-free radial/angular contraction and unchanged private branching
driver. No canonical source edits, advice caches or random model inputs.

## Frozen alternatives and falsifiers
1. Periodic continuation adds an artificial endpoint jump: the nonperiodic
   basis should improve fixed one-sided interval projection at matched9
   angular degrees, and help trained base errors enough to offset cost.
2. Internal cut discontinuities dominate: changing basis can leave/worsen
   trained error despite nicer endpoint controls. Keep the finite negative;
   do not try a degree sweep or select another basis from truth.
3. Primitive normalization is wrong: full interval Gram identity, half-window
   complement/parity, positive-semidefinite bounds, and independently
   integrated entries must pass before trained measurements.
4. Arithmetic cost dominates: GL9 at every radial node costs real basis and
   Gram work. Charge it plus setup and all old work, not just removed complex
   exponentials. Opaque functions remain explicit; no portable total claim.

One fixed basis alternative only. Toy intervals retain fractions .1,.5,.9
of the angular domain; repeated same-cut mass is coefficient norm squared,
with independent analytic Legendre-antiderivative and Fourier coefficients.
Then H2T3/H4T3 base, same models as prior pilot; Q128 diagnostic only.
Freeze expansion gate: both base absolute errors must decrease and their
error²*CPU product must decrease against a same-session cold real-Fourier
run. Otherwise no17-case sweep. This is a finite exploratory gate, not a
population guarantee or sampling-win criterion. Constant/duplicate ties
and positive hidden scaling1e-16 also pass before expansion.

## Status
Protocol frozen before primitive/model measurements. T55 independently
checked normalization, positive GL9 exactness and mechanism limitations.

## Results
Pending; every failure and correction retained.

## Compute accounting
Full cold GL9/radial rule setup, clipped interval construction,9-node angular
basis/Gram per radial node, weighted contractions, branching, geometry and
all named elementary functions. One thread,2GB, shared lock,<=110CPU/cycle.
No new Monte Carlo or training. No degree/basis search.

## Claims
[inferred] The polynomial basis removes periodic identification of angular
endpoints while preserving the same dimension and exact root support.
[speculative] This reduces material finite-projection error efficiently.
Endpoint identification itself is a measure-zero condition, not a proof of
an unavoidable bias; the matched finite representation test is necessary.

## Alternatives not ruled out
Other local/support-aware bases and explicit internal gate boundaries remain
open, as do precise total cost and finite-projection error certification.

## Failure modes
Incorrect angular interval normalization/parity, floating Gram loss of
positivity, radial quadrature error, cost overhead and favorable error
cancellation on the small pilot. DRI owns final claim classification.

## Primitive result
Independent direct angular integrals agree within8.33e-16; full support,
half-window complement/parity, PSD bounds and root identity pass. Independent
Legendre-antiderivative coefficients agree within2.36e-15. At retained
fractions.1/.5/.9, repeated-cut deficits are .0101433/.0186920/.0101433
for the degree8 polynomial versus .0269540/.0248418/.0269540 for Fourier±4.
Thus this fixed toy supports the endpoint mechanism, but says nothing yet
about trained accuracy or efficiency. Primitive cycle CPU.004; full work
recorded in controls.json. No degree was selected from these results.

## Frozen trained pilot: mixed, expansion gate failed

| Model | Reference | Real Fourier | Angular polynomial | Q64/128 change |
| --- | ---: | ---: | ---: | ---: |
| H2T3 base | .410171261727 | .409040821130 | .408387671343 | 1.36e-13 |
| H4T3 base | .984550381359 | .986874860376 | .983124819661 | 1.07e-8 |

Absolute error ratios polynomial/Fourier are1.578 (H2) and.6133 (H4).
CPU changes .0920→.0720 and.3160→.2639 seconds, but ordinary arithmetic
subtotals rise15.63M→40.51M and61.05M→160.92M because explicit positive
angular Gram work replaces opaque Fourier functions. Error²*CPU ratios
are1.949/.3142; error²*ordinary-subtotal ratios6.450/.9913. These partial
cost comparisons do not certify portable total FLOPs. Full pilot including
refinements and six constant/duplicate/scale controls costs1.956CPU seconds.

The frozen both-model gate fails on H2. No17-case expansion or degree search
is run. The endpoint toy gain does not establish a general trained-network
gain; internal projection and error cancellation remain important. Negative
scope is this matched81DOF basis substitution on the two frozen base models.
Independent H2 angular-antiderivative reimplementation requested before
closing the bounded phase. Other local/internal-support bases remain open.

## Finite alternatives checklist
- Formulation: same exact root domain, model, branching and dimension; only
  the angular basis changes. The interval toy directly measures the proposed
  endpoint mechanism, not model accuracy.
- Conditioning/support: exact root support is retained; internal gates are
  still finite projections, so their composition can dominate endpoint gain.
- Representation: one fixed polynomial alternative is tested; mixed/local
  bases and explicit internal support are NOT excluded or selected here.
- Numerical accuracy: positive GL9 is exact-degree for angular products;
  independent scalar quadrature and antiderivatives pass; radial Q128 and
  constant/duplicate/scaling controls do not explain the material H2 loss.
- Cost: removed opaque complex functions coexist with higher counted real
  work. CPU and arithmetic products are both reported; neither is silently
  substituted for a portable complete-cost metric.

No further basis/degree rescue is justified within this frozen pilot: the
predeclared shared-recipe gate failed. A distinct mechanism remains Ready,
while independent reproduction of the H2 loss is the remaining handoff gate.

## Independent handoff
T55 review `1e8d9df` replaces the positive GL9 angular primitive with
Legendre-product antiderivatives while retaining the shared branching and
radial Q64 rule. It returns H2 .4083876713432273, differing by1.11e-16;
identity/complement discrepancies are3.55e-15/2.22e-16, with1.254CPU seconds
including cold setup. This independently reproduces the fixed-basis H2
failure and rules out the GL9 angular integration implementation as its
cause. Shared branching/radial geometry is not independently reimplemented.
The bounded phase is complete. H4's mixed result and other basis/allocation
choices remain open; no harmonic-family rejection or sampling claim.
