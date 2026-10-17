# Exact Gauss-Legendre node and weight enclosures

The task is a prerequisite for floating quadrature qualification, not an integral or cone-mass certificate. Fixed orders6,10,16, root width2^-160 and outward weight grid2^-128 are chosen before results.

Use P0=1, P1=x, kPk=(2k-1)xP(k-1)-(k-1)P(k-2). For x in(-1,1), delete zeros from the sequence P0..Pn and count sign variations V. At a zero of an intermediate Pk, its two neighbors have opposite signs by the recurrence; variation therefore does not change there. At a root of Pn, (1-x²)Pn'=nP(n-1), so the last variation drops by one as x increases. Consecutive Legendre polynomials have no common root by recurrence and P0=1. Thus V(a)-V(b) counts Pn roots for non-root endpoints a,b. Orthogonality gives n simple roots in(-1,1), with reflection symmetry. The even fixed degrees have no zero root. Initial positive domain(0,1) has n/2 roots.

Bisect only intervals with positive count. After exactly160 dyadic levels retain intervals whose endpoint variation difference is1; retain any multiple-root or exact-midpoint-root case as unresolved, without moving the root or changing precision. Distinct intervals and root counts establish coverage, rather than trusting NumPy guesses. Negative roots are exact reflections.

For each root bracket, interval Horner evaluation of the exact derivative polynomial gives a derivative enclosure D. Standard Gaussian-Legendre weights are w=2/[(1-x²)Pn'(x)²]. Exact rational interval multiplication, squaring and reciprocal give raw weight enclosures when the denominator lower bound is positive. Otherwise retain unknown. Round endpoints outward to multiples2^-128 using exact integer divmod. This simplifies later exact summation without losing containment. Every weight is model-independent; no coefficient is learned from target networks.

For consistency, interval moments through degree2n-1 must include their analytic values2/(k+1) for even k and0 for odd k. Odd moments cancel exactly by reflection, even moments use positive powers and interval sums. These checks alone would not prove root identity or an integrand error bound; the Sturm and weight arguments supply the former.

The source binary nodes and weights are ordinary approximations. A source float lying outside a very narrow ideal enclosure is expected, not automatically a numerical defect. Exact distances to the certified ideal intervals bound component errors only. An additional integrand sensitivity and floating evaluation argument is needed before this qualifies an actual source integral. No tensor quadrature, model, mass, geometry, random input or reference integration is part of this task.

Arithmetic is Fraction/integer exact. Named operation counts and peak operands are setup evidence, not hardware FLOPs. Loops, gcd/allocation/control/runtime are not completely priced. Fixed universal advice may be reused under benchmark rule4, but no ranked estimator or PRECOMPUTE_FLOPS claim is introduced.
