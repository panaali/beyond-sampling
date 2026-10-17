# Actual thin Duffy10 truncation verification (T109 continuation)

## Question
Can the already reviewed positive-simplex polynomial bound certify the actual fixed GL10 quadrature on17,955 thin branches of DRI139a7b9e, independently of observed agreement with numerical references?

## Method
Claimebda8f3e before implementation; no overlap T179 finite-source tests or T11579 nativecost. Freeze all observed Duffy branch indices/source hashes; no criterion fitted to errors. Exact original-dyadic Gram: upperb=maxdiag, center lower=min_i(R_i·sumR)^2/||sumR||² if minimum positive; acute lower=mindiag/4 if offdiagonals nonnegative. Choose largest valid lower; reuse only existing132 independently verified exact certificates and recheck KKT on original rays. No new optimization, node sampling or mass replay. All reused certificate construction remains prior separately charged work.

The actual source Duffy coordinates are a cyclic permutation of the prior reference map. For a degreeN polynomial in q=||sum y_i R_i||², pulled-back degrees with Jacobian are at most2N+2,2N+1,2N. GL10 exactly integrates N=8. Exact integral and positive quadrature both have simplex weight1/6; the reviewed uniform Chebyshev tail yields absolute mass error |detR|*tail/(6pi²). Permuting ray coordinates preserves this degree statement; no analytic-inner kernel needed. Degree8 fixed before any results.

## Status
Protocol/primitive prepared; independent source gate before exact controls and trained bounds. No mass integration is executed.

## Results
Pending. Node-only observed minq.3643 is deliberately not used as a global lower bound. Generic small determinant does not imply small mass or fixed-rule accuracy; prior analytic counterexample retained by T179.

## Compute accounting
Exact Fraction operation inventory/maxbits and CPU per cone, MP80 tail work separately; binary integer/gcd/MP/library/control/memory cost not converted to cheap scalar operations. Prior132 certificate setup remains disclosed, not free per-model advice. One thread2GB, <=20CPU review or110CPU primary cycles, resumable256-case records. Saved reference differences used only diagnostic after fixed bounds. No quadrature calls and no source inverse/determinant rerun.

## Claims
[inferred, independently reviewed] Real-arithmetic degree/tail and radial range formulas. Bounds computed with80-digit arithmetic are conditional numerical evaluations, not directed rounding certificates. Actual stored-ray determinant is exact in the bound, while the source's floating determinant, generated GL nodes/weights, geometry and kernel errors are outside it. Coverage is thin branch only; log128 remains separate.

## Alternatives not ruled out
Center/acute ranges may be too loose despite accurate actual quadrature; available KKT certificates can discriminate only132 prior cases. All-new KKT, anisotropic error bounds or another representation remain untested rather than impossible. No bound refinement or cap increase based on failures in this phase.

## Failure modes
Singular rays or absence of a positive proved lower bound stay uncovered; large finite bounds stay explicit and do not imply inaccurate quadrature. No clamping, claim of floating certificate or full estimator accuracy.
