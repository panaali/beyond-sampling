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
