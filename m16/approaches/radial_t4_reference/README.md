# T129 exact radial bounds for positive Gaussian cone references

## Question
Can the remaining56 nonacute T4 error cones be integrated positively with an independently justified radial interval? Claimc081033 preceded implementation.

## Method and frozen protocol
Use exactly the56 T126 uncovered indices and their original dyadic float rays from T110's hash-pinned6347 corpus. No normalizations, rotations or gate enumeration. Form exact rational Gram G=RRᵀ. Enumerate15 nonempty vertex supports S; solve G_S z=1, normalize y=z/sum z, and retain y>=0. Among candidates choose minimum a=yᵀGy. Verify exact global KKT: (Gy)_i>=a with equality on positive support. This gives a>0 for independent rays. Convexity gives b=max_i G_ii. Save exact rational a,b,detR,y and slacks so another implementation can verify them without optimization.

For q in[a,b], set alpha=(a+b)/2, delta=sqrt(ab), rho=(sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a)). The Chebyshev coefficients of q^-2 have absolute magnitudes2(alpha+k delta)rho^k/delta³ for k>=1. The uniform tail after degreeN is
`2*rho^(N+1)/delta³ * [alpha/(1-rho)+delta*((N+1)/(1-rho)+rho/(1-rho)^2)]`.
Fixed tensor GLn under the Duffy map is exact on the degreeN=n-2 truncation. Its positive Gaussian mass error is at most abs(detR)*tail/(6pi²). T55 independently checked KKT enumeration, coefficients and degree argument. Bound evaluation uses50-digit arithmetic from exact rational inputs; this is an integration-only expression, not directed floating certification.

First check analytic positive-correlation block orthants: correlations a,b in{0,.4,.9}, base/uniform factor8/fixed coordinate permutation,27 cases. Their probabilities are products of bivariate orthant probabilities; test fixedGL20 against its explicit bound plus5e-14 floating allowance. Verify minimum certificates exactly; each scaling must scale radial bounds by64 and determinant by4096 while preserving mass and probability bound. Include exact diagonal Gram control q_min=1/4,b=1 and the rho=0 constant-q tail branch.

For each of the56 trained cones, compute exact bounds first. Select the first n in{20,24,28,32,40,48,64} whose50-digit integration bound is<=1e-9/56. If none qualifies use64 as an explicit insufficient-bound diagnostic; never silently call it certified. Selection depends only on exact geometry and the fixed target, not author or reference values. Preserve every selected order, bound, finite failure, setup cost, and all source differences. Partial checkpoint each8 cones; primary<=110CPU hard/90CPU soft,2GB/one thread; smaller20CPU review phase allowed. No nearest expansion. Combine with T123/T126 only if all cones have valid references, retaining arithmetic/geometry qualifications and summed conditional bounds.

## Alternatives and falsifiers
Source failure could reflect positive-reference error: independent analytic obtuse controls and exact KKT certificates test the new representation. Bounds may be too loose for fixed finite orders: record that limit, do not tune after seeing accuracy. Fraction setup may cost more than integration: separately time it. Floating determinant/node/accumulation errors remain outside the analytic quadrature bound. General interval splitting, more accurate arithmetic, or source endpoint repairs remain open if this bounded phase is inconclusive.

## Status / results
Protocol frozen; no measurements yet.

## Compute accounting
Record exact geometry setup,50-digit bound/order selection, GL construction/integration, and known39n³ integrand arithmetic separately. Full CPU includes implemented setup and writes; complete portable elementary cost is not claimed. Previous phases' costs remain charged separately for a complete reference.

## Claims / failure modes
[inferred] Exact radial bounds permit a positive representation of any independent pointed simplicial cone.
[speculative] The fixed finite order set makes all56 remaining references useful. No cheap estimator, floating certificate or solved-rung claim. Singular inputs and near-rho1 expensive bounds need explicit rejection/guarding; never clip an inaccurate negative mass to zero.

## Results and bounded completion
All27 analytic controls and exact special cases pass. All56 trained radial certificates meet the frozen error target withGL20, no higher-order trial required. Their mass .026630168417774144, summed integration-only bound5.382e-14, source-minus-positive5.263e-9; phase .133CPU. Combined6347-cone positive reference gives .4770131259941194, differing3.89e-16 from the independent whole-model reference, summed conditional quadrature bound3.158e-8 and zero negative values. End-to-end floating accuracy and portable cost remain uncertified. Total known positive-integrand arithmetic across all phases1.980264B; no cheap-estimator claim. [HANDOFF.md](HANDOFF.md) preserves qualifications and reproduction details.
