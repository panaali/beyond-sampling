# T143 positive analytic inner mass

## Question
Can exact integration of the last Duffy coordinate reduce the positive Gaussian cone reference's n³ tensor cost while retaining its real-arithmetic error guarantee? Claim71d3d06 was pushed before implementation. T144 determinant intervals, T142 Plackett attribution and T140 fixed-rule replay are separately owned.

## Method / frozen protocol
Use the same original6347 quadratic T110 rays and T130 per-cone/aggregate selected orders and tolerances. Do not change geometry, bounds, node choices, interval shortcuts or unresolved cases. First independent analytic scalar integral controls and27 block-normal cone controls, then fixed16 prior indices atn20. Only after controls pass, replay previous T130 quadrature rows under the exact prior orders. Preserve raw discrepancies, negative/nonfinite or cap failures, all costs and independent review. No new model, noise or truth.

For q(z)=a z²+2bz+c=||P+zD||², δ=ac−b²>0, θi=atan((azi+b)/sqrtδ), Δ=θ1−θ0=atan2(sqrtδ,c+b). The integral is a/(2δ^(3/2))*[(Δ−sinΔ)+sinΔ*(1+cos(θ0+θ1))]. Each bracket term is nonnegative. Evaluate1+cosΣ as[(cosθ0+cosθ1)²+(sinθ0−sinθ1)²]/2; compute δ as sum of six squared exterior products. For Δ<=1/8 use the fixed alternating polynomial Δ³/3!−Δ⁵/5!+Δ⁷/7!−Δ⁹/9!+Δ¹¹/11!, with real-arithmetic absolute remainder<=Δ¹³/13!. Otherwise direct Δ−sinΔ. Do not change thresholds after trained results.

Freeze fallback to same-order positive GLn in z if a*c<=0, δ/(a*c)<1e−12, nonfinite or nonpositive formula, invalid angle, or other explicit unsupported condition. Exact a=0 uses constant integrand. Both exact-inner and GLn integrate degree<=2N z-polynomials exactly forN=n−2. Outer Duffy degrees<=2N+2 and2N+1 remain within GLn, so the old positive-functional Chebyshev error bound det*tail/(6π²) applies in real arithmetic to hybrid node-wise choice. T55 independently derives this proof and the identity. Finite analytic series error and floating errors are separate; no floating certificate is claimed. Series truncation error is bounded and reported with its positive outer weight.

Alternative explanations: numerical cancellation may offset node savings (fixed near-collinear/scaled/obtuse scalar controls against80digit integration); geometry/determinant drift may explain changed output (same rays, retained prior determinants/orders and full per-cone parity metadata). Cost could shift to transcendental/setup work (explicit scalar arithmetic/special/compare counts and cold node plans; unknown Legendre/library/hardware costs stay stated). Positive output alone does not certify relative precision.

## Status
Bounded phase complete;181 supported controls pass, all saved6347 rays replayed at both unchanged T130 objectives. Corrected named totals31,692,407/94,913,850 reduce prior work64.6%/74.0%; same32 strict-objective unresolved. No self-confirmed claim. See HANDOFF.md and cost_correction.json.

## Compute accounting
One thread,2GB/process, primary lock<=110CPU or review<=20CPU. Count written scalar array expressions/reductions/special calls/comparisons in the new kernel, cold Duffy plans and all fallback calls. Retain inherited T130 setup/selection costs separately; deterministic geometry is not free. Model-independent constants are declared precomputed; implicit loop/memory and hardware/library internals excluded. Fresh process/cold plans per objective. No competitive estimator claim from local kernel savings alone.

## Claims and failure modes
[speculative] An analytic inner primitive can lower the cost of the same positive reference. Bounds inherited from T130 exclude floating geometry and determinant arithmetic; T144 may separately improve that factor after independent handoff. Nearly collinear rays, underflow, finite quadrature limits and roundoff remain distinct possible failures, not representation impossibility. Untried alternatives include other coordinate orderings, stable series around a midpoint, adaptive positive outer rules and lower-bound improvements. None is retired.

Independent recover_56 source review identified omitted explicit Boolean masks/reductions and divergent standalone inner domains. Initial results/source are preserved;40e4346 corrects every charge algebraically without rerunning trained kernels, validates124 controlledgers and preserves181 supported values exactly. Standalone zero endpoints/origin crossings are explicitly rejected. API requires nonempty node arrays (n>=2); empty-array reductions are outside this scope. Indexing/flatnonzero memory scans remain excluded. Final T55 raw audit pending.
