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
Frozen bounded phase complete; independent T55 source/certificate/raw review bebac70 passes.

## Results
All32 prior failures return at the fixed order cap:13 use the acute Gram lower bound,19 use exact KKT after a negative off-diagonal sign. Orders12/16/20/24/32 occur7/12/1/11/1 times. Lower bounds improve1.6064–6.3491 times over the old center-based bound. No geometry or integration-kernel change was needed. All6315 prior values are bitwise unchanged.

The combined ERROR probability is.5229868745335334, hence accuracy.47701312546646657 versus independent reference.4770131259941194 (difference−5.2765281393e-10). Combined conditional bound1.6087896903e-8; reference conditional bound3.1579959186e-8. New32 bound sum9.7996982667e-11, maximum new absolute difference from reference1.1764091966e-13. The32 determinant-factor corrections are included; inherited6315 assumptions remain. This is not full floating certification.

All48 exact radial controls pass their known minima;24 return analytic-mass estimates within the stated conditional comparisons and24 remain unresolved at the frozen order cap. These unresolved controls are preserved. MP80 versusMP100 selected-bound rounded displays agree for all32; this is a precision check, not outward certification.

The first summary mislabeled summed error masses as accuracy. Both initial JSON and summary script are preserved as summary_initial_label_error.json and summarize_initial_label_error.py. Corrected summary explicitly reports both probabilities; combined archive is byte-identical (SHAe1d98a0bc0084fe154ff22cb305b173a594b164d46ba25aed0eeb8e16f77cb58). No numerical rerun or changed estimates.

Independent review bebac70 reconstructs all32 original-ray exact Gram/determinant certificates,13 acute and19 KKT certificates,100-digit bound/order decisions, all6315 retained values/bounds and the2,038,604 new named kernel count (.120CPU). It confirms the corrected probability semantics and separately qualified exact/MP cost; it does not certify excluded general floating arithmetic.

## Compute accounting
One thread,2GB, review lock<=20CPU per process. All exact per-model certificate work is charged by CPU and rational operation/bitwidth inventory, never universal free advice. Big-integer implementation cost is not converted to scalar FLOPs. Baseline T14394,913,850 named operations includes prior failed attempts; new kernel/selection/radial work is additional. High-precision bound arithmetic, Fraction gcd/division internals and library/memory costs retain explicit unknown portable cost. CPU includes wrapper overhead and persistence; final write/return excluded. No model forwards or Gaussian sampling.

Four trained chunks total1.02118939CPU including repeated imports; controls.345072878CPU. Exact certificates.129326029CPU, high-precision selection.037603321CPU, kernel.027387328CPU are included components. New kernel2,038,604 named operations plus baseline94,913,850 gives96,952,454 known named operations, excluding explicitly inventoried exact/MP internals. Four cold rule plans add30,044 diagnostic arithmetic operations; they are not four mandatory setups for a single cached call. Rational operands/results reach799 numerator/796 denominator bits; full operation categories are saved in summary.json. This remains an incomplete portable cost, not a speed or challenge-win claim.

## Claims
[measured] All32 fixed failures close using sharper exact radial lower bounds without changing cones or the order cap. [inferred] Their previous unresolved status was not evidence that this positive representation was exhausted. No self-confirmed, cheap-estimator or full floating certificate claim.

## Alternatives not ruled out
Ray rescaling, deterministic edge subdivisions, adaptive positive outer quadrature and different error allocation. No failed finite bound establishes representation impossibility. Improved bound coverage alone does not imply cheap inference or full floating certification.

## Failure modes
Exact rational setup may dominate cost. A known real-arithmetic tail bound does not bound all binary64 kernel operations. Input ray rounding, old determinant discrepancy and analytic series truncation are distinct; only the latter two receive the explicitly stated factor/remainder treatment here.
