# T167 frozen positive-mass transfer to H8T4 seed1

## Question
Does the independently reviewed positive selector transfer unchanged from seed0 to the existing held-out seed1 error geometry? TASKS-only claim49189965 precedes implementation. T109 fullseed0 review ea5d57df passes; T148 seed1 source/positive references are already independently reviewed. No new truth, geometry, inputs or estimator tuning.

## Method / frozen protocol
Use only frozen_recipe_holdout/exact/error_captured.npz (SHA215ce37b68c283e29224a2c720ae30c24aaf741b1269bf6bc5d0f2044b6ec9db),30726 original rays, and its eight positive reference batches. Keep T161 source selector, kernel, center determinant and exact acute/KKT fallback unchanged. The sole dataset parameter is N=30726, giving tolerance1e-7/N in both floating center and exact-rational selector. Orders2,3,4,6,8,12,16,20,24,32 unchanged. First64 indices floor(k*(N-1)/63),k0..63; validate metadata/source and first qualifying-order/target consistency, then require independent gate before all remaining original rays. Existing analytic controls are reused because the algorithm is unchanged; source/hash and tolerance plumbing are checked explicitly.

Alternative explanations: performance could rely on seed0 cone shapes (fixed new-checkpoint64 then full census); a tolerance mismatch between float and Fraction selectors could cause false coverage (assert both derive from sameN); exceptional cones could fail numerical positivity (preserve and stop on failure/reference discrepancy). Finite-order unresolved cases stay unresolved. No candidate substitution into frozen T14924/96 and no model-dependent tuning or favorable-index selection.

## Status
Bounded full transfer complete; independent full source/raw/cost audit passed88490d79. Pilot gate independently passed2827c813; TASKS14983b7b froze full phase before implementation.

## Results
All30726 return, with64 pilot rows reused once. Accuracy .9874600098992554 versus independently frozen reference .9874600108260589, difference−9.26803523e-10. Sum conditional integration bounds1.14093936608e-8, reference-error L1 1.55760650188e-9, maximum cone discrepancy2.53238447e-12. No unresolved or failed rows. Methods:16420 center intervals,14251 center quadratures,55 exact KKT rescues. No trained geometry/reference rerun.

## Compute accounting
Same detailed named floating/category counters, exact rational/MP operation inventories, cold plans and failed trials as T161/T109; unknown library/gcd/MP, integer/control/memory and FP error remain explicit. One thread2GB, <=20CPU review cycles; no completed reference or geometry replay. No full-cost/noisy-rung or challenge claim.

## Claims
[inferred, independently reviewed] Unchanged positive kernel and real-arithmetic bound. [measured] The unchanged algorithm covers this second captured corpus at the frozen conditional target. [measured] The300-operation goal is unmet:3323.66 known units/cone. Neither statement certifies arbitrary floating error or complete model-facing cost.

## Alternatives not ruled out / failure modes
A failed finite-order or kernel test is scoped and preserved, not a general impossibility. The original<300operations-per-cone goal remains unmet on seed0; this transfer tests accuracy/cost stability, not that target by assumption. Full results require a separate frozen phase and independent audit.

Initial preflight (before any cone evaluation) caught an overly strict equality assertion: binary64 `(1e-7)/30726` is one ulp below the rounded exact Fraction tolerance. `pilot_initial.py` and `preflight_failure.json` preserve this harness issue. The estimator arithmetic and target are unchanged; validation now proves via Fraction that the floating tolerance is conservative and differs by at most one ulp. No target relaxation or numerical result replacement.

Fixed64 transfer result: all64 returned,35 center intervals and29 center quadratures, no exact rescue. Maximum independent-reference discrepancy3.03400e-13; summed absolute discrepancy1.09269e-12 versus conditional bound2.10497e-11. Known floating/category total532,864 plus unknown library/FP terms, CPU0.195270. This selected pilot has a larger cost than seed0's pilot; no corpus extrapolation is made. All failures and original preflight issue remain preserved. Independent source/tolerance/raw gate requested before full transfer.

Reproduce fixed64 only if needed using one-thread2GB review-lock wrapper and `/workspace/clean-neurons/.venv/bin/python m16/approaches/positive_h8_holdout/pilot.py`. Completed results are ready for saved-data review without rerunning integration.

Full source and data: `full_protocol.json`, `full_run.py`, `full_summarize.py`, `shard_000..120.json.gz`, three uniquely named session records, `summary.json`, `values.npz`. Each256-index shard preserves all trials, setup, selected orders, exact certificates, comparison and ledger. The final shard has6 rows. Attempts have unique names so resumed CPU would not overwrite prior attempts.

Known numerical/category subtotal102,122,861, including one canonical cold rule plan union18,050 and six universal constants. Actual staged cold-plan work65,236 is separately recorded; total staged CPU including pilot/imports11.996909, aggregation0.463958. Exact setup55 rescues has maximum793/770 numerator/denominator bits, certificateCPU0.224280 and MPselection0.067847; full operation inventory retained. GCD/MP/Legendre internals, integer/control/memory and capture/geometry are outside the named subtotal. T171 explicitly tracks model-facing capture/N/storage integration before adoption.

Original commands (do not rerun completed cones): one-thread memory-limited review-lock Python `full_run.py 0 32`, `full_run.py 32 72`, `full_run.py 72 121`; saved-data aggregation `full_summarize.py`. Existing completed shards are skipped and partial shards resume only missing suffixes. No source fallback, extra order, changed target or reweighted selection was introduced.

T55 independent full audit88490d79 verifies all30726 indices/reference rows/shard hashes,64 pilot rows bitwise once, all55 exact original-ray Gram/KKT/determinants and80-digit selected bounds, exact stored-float error9.26803537875e-10 and both canonical/staged rule ledgers. Reviewer0.848CPU, no integrations rerun. All cost/FP/capture qualifications remain.
