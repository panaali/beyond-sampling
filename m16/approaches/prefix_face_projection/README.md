# T164: unbounded crossing-face projection

## Question
Does T159 local negative-prefix invariance extend to weight-derived sectors without truncating either remaining Gaussian input?

## Method
Freeze the same H2T4seed0 weights, label2, positive target normalized to u=1, negative prefix x0<0 giving h1=(s,0),s>=0. Since prefix<target, need1 has rank sides x<=1<=y or y<=1<=x. Enumerate EVERY original moving boundary from T147, not T155 locally feasible words. Reuse independently checked crossing-face constraints: full correctness/final support for readout margins, conservative first/middle/rank constraints for final gates. Substitute boundary equality when its input normal is nonzero, then Fourier–Motzkin eliminate the remaining input. Exact rational projected intervals include lines/points and unbounded endpoints. Union every occurrence/side; open complement components are sufficient invariance intervals. Zero input normal with nonzero state derivative stays unknown. Zero state derivative contributes no moving boundary.

## Status
Claim3263e8a2 pushed before implementation. T55 projection/invariance theorem reviewed; primitive and raw checks pending.

## Results
Predeclared controls: empty, all-state, bounded and unbounded projections, singleton, zero coefficient, positive row rescaling and synthetic equality substitution. Independent projection/dual checks before interpreting trained output. Only negative prefix / positive target sector in this phase.

## Compute accounting
No model forwards, Monte Carlo, pair integrals or interpolation. Exact finite coefficient arithmetic/projection and one checkpoint load charged. One thread2GB, <=20CPU review cycles.

## Claims
[inferred] Absence of relevant moving correctness boundaries on a connected projected complement implies pointwise a.e. prefix invariance on that sector. This is sufficient, not maximal actual invariance. Constants, Gaussian mass and estimator integration remain separate.

## Alternatives not ruled out
1. Local invariance could rely entirely on the [-6,6] cutoff: unbounded projection is the discriminator.
2. Broad final-gate faces can hide true invariance: a covered s interval is inconclusive, not proof of variation.
3. Other signs/rank sectors, direct integration, and state-weighted summaries remain open; no whole-method negative.

## Failure modes
Dropping unbounded rays or point projections; reusing local word exclusions; treating a covered interval as actual output variation; ignoring state-dependent zero input normals; or calling sufficient intervals maximal invariance sectors. Homogeneity only rescales positive u; u<0 is not tested.
