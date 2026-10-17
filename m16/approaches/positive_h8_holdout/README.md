# T167 frozen positive-mass transfer to H8T4 seed1

## Question
Does the independently reviewed positive selector transfer unchanged from seed0 to the existing held-out seed1 error geometry? TASKS-only claim49189965 precedes implementation. T109 fullseed0 review ea5d57df passes; T148 seed1 source/positive references are already independently reviewed. No new truth, geometry, inputs or estimator tuning.

## Method / frozen protocol
Use only frozen_recipe_holdout/exact/error_captured.npz (SHA215ce37b68c283e29224a2c720ae30c24aaf741b1269bf6bc5d0f2044b6ec9db),30726 original rays, and its eight positive reference batches. Keep T161 source selector, kernel, center determinant and exact acute/KKT fallback unchanged. The sole dataset parameter is N=30726, giving tolerance1e-7/N in both floating center and exact-rational selector. Orders2,3,4,6,8,12,16,20,24,32 unchanged. First64 indices floor(k*(N-1)/63),k0..63; validate metadata/source and first qualifying-order/target consistency, then require independent gate before all remaining original rays. Existing analytic controls are reused because the algorithm is unchanged; source/hash and tolerance plumbing are checked explicitly.

Alternative explanations: performance could rely on seed0 cone shapes (fixed new-checkpoint64 then full census); a tolerance mismatch between float and Fraction selectors could cause false coverage (assert both derive from sameN); exceptional cones could fail numerical positivity (preserve and stop on failure/reference discrepancy). Finite-order unresolved cases stay unresolved. No candidate substitution into frozen T14924/96 and no model-dependent tuning or favorable-index selection.

## Status
Frozen fixed64 transfer phase, implementation pending.

## Results
Pending. Prior seed1 exact positive accuracy .9874600108260589 is reference only, not a fitted target.

## Compute accounting
Same detailed named floating/category counters, exact rational/MP operation inventories, cold plans and failed trials as T161/T109; unknown library/gcd/MP, integer/control/memory and FP error remain explicit. One thread2GB, <=20CPU review cycles; no completed reference or geometry replay. No full-cost/noisy-rung or challenge claim.

## Claims
[inferred, independently reviewed] Unchanged positive kernel and real-arithmetic bound. [speculative] Transfer coverage and useful cost remain to be checked on seed1.

## Alternatives not ruled out / failure modes
A failed finite-order or kernel test is scoped and preserved, not a general impossibility. The original<300operations-per-cone goal remains unmet on seed0; this transfer tests accuracy/cost stability, not that target by assumption. Full results require a separate frozen phase and independent audit.
