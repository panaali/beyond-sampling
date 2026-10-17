# T152: radial-bound discriminator on32 unresolved cones

## Question
Are T130/T143's32 aggregate-target failures caused by loose radial lower bounds rather than an exhausted positive representation? Claim74ac4cf and pre-implementation fallback scope01de7eb were pushed first. T143 kernel/bound independently reviewed851e2e6; no shared edits.

## Method
Freeze original6347 rays SHA07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581 and precisely the32 unresolved indices in T143 aggregate.json SHA88f1969fbcb20f95e21fafc4332eb5ab6f153d88ed8f210d5ddddf4b40309488. Existing6315 values/bounds SHA879d18f30ee4dfdd1608c2e03bcccdf1e17b2b0af51f81ba42cc988de7464e13 remain unchanged. Pin corrected analytic-inner primitive SHA30919d71e286ccc707b5acac5fbb5fa9d9a75b15d03d94b7d10267e37e4b0542 and reviewed exact radial helper SHA067e21bddd4c895653e0f3242a9521c7f05650671fc3bd8d256b4d433169e76b. No normalized/new rays, geometry enumeration, other models, random inputs, noise or source-order tuning.

For exact binary ray Gram matrix, if all6 off-diagonal dot products are nonnegative, first try a=min_i||r_i||²/4, b=max_i||r_i||². This follows from sum(y_i²)>=1/4 on the simplex. Otherwise use the exact15-support constrained-quadratic KKT minimum. If the acute bound still fails the frozen order list, try that same KKT minimum as the predeclared rescue. Preserve both attempts. Source old95 determinant remains the integration coefficient, taken from the already charged failed T130 pass; no T144 point-formula substitution.

Use SAME orders{2,3,4,6,8,12,16,20,24,32} and exact tolerance1/(10^7*6347). Evaluate the reviewed Chebyshev bound with exact a,b and exact determinant at80 decimal digits, plus the explicit determinant coefficient discrepancy term |d_old-d_exact|/(12π²a²). The latter bounds the factor error through any positive constant-exact rule on the simplex. Pick the first qualifying order. Evaluate only that order with the pinned T143 kernel, then add its reported series remainder; retain unresolved if the final total exceeds the target. High-precision bound displays are not outward arithmetic certificates; all other floating geometry/kernel/rule errors remain excluded. No outputs/reference values influence order selection.

Before trained work freeze48 radial controls: identity, positive diagonal(1,2,3,4), nonacute block rays e0,(-2,1,0,0),e2,e3, and boundary-minimum rays e0,(2,1,0,0),(2,0,1,0),(2,0,0,1), each under three global dyadic scales2^-5,1,2^5 and four cyclic row orders. Exact known minima are1/4,144/205,1/12,1 respectively (scaled byscale²). Check every exact certificate and acute lower bound. The first three families also have analytic Gaussian masses1/16,1/16,atan2(1,-2)/(8π); compare only returned finite-order estimates, preserve unresolved controls. This is a bounded reference discriminator, not a cheap-estimator claim.

Process trained32 in four fixed8-case chunks<=20CPU. Save full Gram/certificate/selection history and rational arithmetic counts/operand bitwidths, CPU and all kernel/cold-rule work. If all return, merge their values with existing6315 and report conditional bound sum and deviation from independently established .4770131259941194 reference, without treating its own conditional error as zero. Any remaining failure preserves deterministic edge-split/alternate representation as an explicit next option, not a blanket negative.

## Status
Protocol frozen; counted exact wrapper and analytic controls next.

## Results
None yet.

## Compute accounting
One thread,2GB, review lock<=20CPU per process. All exact per-model certificate work is charged by CPU and rational operation/bitwidth inventory, never universal free advice. Big-integer implementation cost is not converted to scalar FLOPs. Baseline T14394,913,850 named operations includes prior failed attempts; new kernel/selection/radial work is additional. High-precision bound arithmetic, Fraction gcd/division internals and library/memory costs retain explicit unknown portable cost. CPU includes wrapper overhead and persistence; final write/return excluded. No model forwards or Gaussian sampling.

## Claims
[inferred] Exact sharper radial bounds can distinguish a loose-bound failure from genuine finite-order insufficiency. [speculative] The fixed32 cases may close without changing cones or order cap.

## Alternatives not ruled out
Ray rescaling, deterministic edge subdivisions, adaptive positive outer quadrature and different error allocation. No failed finite bound establishes representation impossibility. Improved bound coverage alone does not imply cheap inference or full floating certification.

## Failure modes
Exact rational setup may dominate cost. A known real-arithmetic tail bound does not bound all binary64 kernel operations. Input ray rounding, old determinant discrepancy and analytic series truncation are distinct; only the latter two receive the explicitly stated factor/remainder treatment here.
