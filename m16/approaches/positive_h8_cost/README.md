# T161 positive analytic-inner H8T4 cost discriminator

## Question
Can the reviewed positive2D analytic-inner primitive with cheap radial bounds supply accurate masses at useful cost on in-regime H8T4 seed0 captured error cones? TASKS-only claimbabe21f8 was pushed first. This is a fixed64 pilot, distinct from T160 Schur polynomial and the frozen T149 candidate.

## Method / frozen protocol
Use unchanged T13135098 error simplices (`leaf_mass_review/error_captured.npz`, SHA15f9e5e483c69325c10fcbe4eaaa80f91503b177070ccec1ff2569a11c9436b6). Select exactly floor(k*35097/63), k0..63, before reading per-index masses. Existing eight T131 positive reference archives supply comparisons; no new truth, model evaluation or geometry. Allocate aggregate target1e-7 uniformly, tolerance1e-7/35098 per cone.

Start with T130 center bound and95-operation determinant, preserving the original rays. If the mass midpoint interval fits, return it; otherwise choose the first order in2,3,4,6,8,12,16,20,24,32 satisfying the reviewed Chebyshev real-arithmetic integration bound, then evaluate the corrected T143 analytic-inner kernel. If center geometry/order fails, use T152 exact acute bound and then exact15-support KKT if needed, with the same orders/target and determinant discrepancy allowance. Every failed trial/setup and exact/MP operation inventory remains charged separately. Positive inner kernel guards or unresolved bounds are explicit failures, without source fallback, clamping or quiet omission.

Universal π factors are precomputed once (six arithmetic units); geometry/selection counters and kernel/rule counts are separate, with unknown Legendre/gcd/MP internals and general floating errors retained. Known named units are not complete native/hardware cost. Existing source controls are reused; the new selector gets12 analytic fixtures: orthant, diagonal, nonacute wedge, boundary-minimum cone, each scaled2^-5,1,2^5. Known analytic masses checked where available. New selector controls must pass before the fixed trained64.

Alternative explanations and discriminators: cheap radial center bounds may waste many orders (record failed center choices and exact-rescue improvement); analytic integration may suffer cancellation on H8 thin cones (compare unchanged independently positive T131 references and preserve all failures); apparent low cost could omit per-cone setup (keep counters/exact rational bit widths and CPU). No favorable-case selection or sample extrapolation to full corpus. Larger corpus requires independently reviewed pilot and a separately published bounded phase.

## Status
Protocol frozen; implementation/controls pending.

## Results
Pending. All finite-order/kernel failures will be retained.

## Compute accounting
One thread2GB, <=20CPU review controls/pilot with20CPU cap (preserve partial if18CPU reached), primary <=110CPU only if separately scoped later. Selection, failed attempts, exact certificate setup, numerical kernel, cold rules and import/IO CPU are distinct. No general floating-point or complete-cost certificate.

## Claims
[inferred, reviewed in prior tasks] Positive analytic-inner real-arithmetic bound transfer and exact acute/KKT radial bounds. [speculative] Adaptive low orders can reduce in-regime mass cost; pilot must discriminate this.

## Alternatives not ruled out / failure modes
Finite-order exhaustion is inconclusive, not impossibility. Better representations, subdivision or stronger bounds remain open. Existing references are conditional on captured floating geometry and arithmetic. This pilot alone establishes no full accuracy, sampling advantage, noisy-model generality or challenge solution.

## Preserved initial control failure

Before any trained evaluation, the unequal diagonal control correctly returned `unresolved` at the frozen order32 cap: its exact KKT bound1.38192994e-5 exceeds the2.849e-12 target. The initial harness incorrectly required every control to return a value. `run_initial.py` and `controls_initial_unresolved.json` preserve this harness failure. The algorithm, target, geometry and orders are unchanged. Corrected controls accept explicitly unresolved status, independently evaluate the rejected order32 candidate against the known analytic mass using its saved looser bound, and keep this validation work separate from estimator output/counts. Returned controls must still meet the frozen target; scale invariance must preserve status and value. This is finite-bound inadequacy for unnormalized diagonal rays, not an analytic-inner kernel failure or broad impossibility.

## Fixed64 pilot result

All64 frozen indices returned a mass within the per-cone target; no failures or omitted cases. Modes:29 center intervals,34 center quadratures (24 at order3,1 at4,7 at6,2 at8), one exact-KKT rescue (index31197, order12). Maximum discrepancy from existing independent references4.79456e-13; sum absolute discrepancy2.74928e-12; conditional bound sum2.77228e-11. This is the selected64's mass diagnostic, not a full accuracy estimate or global cost extrapolation.

Named float/category total147,062 includes failed center trials, kernel, cold rule transformations and six universal constant operations. The sole exact-rescue certificate has maximum numerator/denominator731/718bits; its rational operation inventory and MP selection CPU are explicitly outside that float total. Pilot CPU including imports/checks0.217767; controls0.117095CPU. Existing rule/certificate/source qualifications remain. No broad failure or full-cost claim.

Reproduce saved summary: `/workspace/clean-neurons/.venv/bin/python m16/approaches/positive_h8_cost/summarize.py`. Original controls/pilot commands use `flock /workspace/work/codex-review.lock bash -c 'ulimit -v 2500000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/positive_h8_cost/run.py controls'` (replace controls with pilot for the fixed64). Completed experiments need not be repeated for review.

Status: bounded pilot complete, independent source/raw gate pending before any larger corpus. Parent/DRI registry requested. The synthetic unequal diagonal/nonacute unresolved cases remain preserved and separately checked; no target/order relaxation was used.
