# T235 cheap tie-height face-family prerequisite

## Question
Can the simple label(kind,ub) supply model-independent family probabilities and conditional face construction, without the network-dependent wt/flips fields of the earlier coarse label?

## Method
Source/proof only under claim112cee94, after independently completeT231. Freeze four T218 files at60967f89 and the reviewed T230 design. THEORY.md distinguishes the ideal face law, exact-real interpolation of stored grid values, and the finite numerical/Sobol program. Source inventory in SOURCE.md. Independent reviewer recover_910 in review/t235_face_law_review.

## Status
Frozen theory/source proposal awaiting independent review. No executable candidate or controls, no grid/array/model/face/RNG/mass calls. Any implementation, numerical controls or target use needs a separately scoped gate; new target work requires DRI priority.

## Results
No numerical result. The source cuts ub at0,.25,.5 using the tied coordinate u. A label using only(kind,ub) is available before the network walk, including zero-output faces which the current tail-label archive omits. Ideal probabilities are ratios of one-dimensional integrals. For a valid fixed monotone CDF grid, the real-arithmetic interpolated distribution has explicit piecewise-linear bin masses and admits a conditional latent-uniform map. Neither statement establishes useful residual variance or exact probabilities for finite binary arithmetic or a fixed Sobol design.

## Compute accounting
Git reads, byte copies/hash and source/theorem inspection only. Zero author imports, numerical integrations, grid construction, NPZ decoding, face generation, network forwards, fitting, RNG or Monte Carlo. A future implementation must charge probability setup, conditional mapping, label evaluation, new integrand evaluations, preprocessing and any fitting; model-independent does not mean cost-free.

## Claims
- [inferred, source] Unlike wt/flips, ub depends only on the tie value before the network walk.
- [inferred, theorem] Ideal conditional probabilities and real grid-law conditional mapping exist under the stated assumptions.
- [speculative] These cheaper families may help deterministic quadrature or a properly counted sampling/control-variate baseline. No performance, ARC eligibility or adoption claim follows.

## Alternatives not ruled out
Within-bin integrands may remain too irregular/heavy-tailed; a cheaper label can lose predictive information. Numerical inverse-CDF and walker errors may dominate. Full-family data, held-out prediction, probability errors and source cost must be assessed before any gain claim. Conditional deterministic quadrature and stochastic stratification have different requirements; random sampling does not become deductive by using bins.

## Failure modes
Nonfinite/nonmonotone CDF grids, zero normalization, zero-probability bins and T<3 kind handling need explicit rejection/qualification. Endpoint conventions are measure-zero under continuous uniforms but can affect fixed finite rules. Source grid normalization differs from sourceZ; fixed Sobol points do not carry automatic iid probabilities. Conditional clipping/tau8.5 and walker truncation remain inherited approximation issues.
