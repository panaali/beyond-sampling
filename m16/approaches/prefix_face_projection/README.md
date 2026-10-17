# T164: unbounded crossing-face projection

## Question
Does T159 local negative-prefix invariance extend to weight-derived sectors without truncating either remaining Gaussian input?

## Method
Freeze the same H2T4seed0 weights, label2, positive target normalized to u=1, negative prefix x0<0 giving h1=(s,0),s>=0. Since prefix<target, need1 has rank sides x<=1<=y or y<=1<=x. Enumerate EVERY original moving boundary from T147, not T155 locally feasible words. Reuse independently checked crossing-face constraints: full correctness/final support for readout margins, conservative first/middle/rank constraints for final gates. Substitute boundary equality when its input normal is nonzero, then Fourier–Motzkin eliminate the remaining input. Exact rational projected intervals include lines/points and unbounded endpoints. Union every occurrence/side; open complement components are sufficient invariance intervals. Zero input normal with nonzero state derivative stays unknown. Zero state derivative contributes no moving boundary.

## Status
Claim3263e8a2 pushed before implementation. T55 projection and application source reviewed before trained execution. All finite controls pass; independent review9558b53e passes all198 projections and3296 nonnegative dual rows, complete pair generation and exact lifted witnesses. Shared board later reused T164 for an unrelated estT_fix task; this task is unambiguously prefix_face_projection, and root was notified for DRI renumbering.

## Results
Nine exact projection/scaling controls, three equality/rank substitutions and three interval-union/complement controls pass. CPU clock recorded0 for each of two tiny control runs; this is clock resolution, not free computation.

Frozen result:198 moving original occurrence/rank-side projections,14 feasible, no unknown normal; their closed union is[0,+infinity). Thus this sufficient test returns no global unbounded-input invariance interval. It does NOT establish actual variation or failure of state compression. All14 feasible faces use first gates(OFF,ON) and rank side10. Their s=1 witnesses require x about299–300, explaining how finite-cutoff local invariance can coexist with full projection coverage. No new model inputs or pair values were evaluated; these are exact affine-face witnesses. Independent raw review9558b53e passes; reviewer sentinel-order failure preserved in their report.

Reproduce `python controls.py` then `python pilot.py` with the project interpreter, one thread, review lock,2GB/20CPU limits. Trained exact projection time.187657019CPU, excluding imports;1 checkpoint load,0 model forward/MC/integration.

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
