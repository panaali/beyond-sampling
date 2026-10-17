# T104: fixed tangent frames for six-input triangle masses

## Question
Can the existing guarded analytic triangle integral reuse a cone's fixed tangent plane across its outer quadrature nodes, reducing actual integration work without changing the formula, tolerance or fallback policy?
Claim published a1d6a50 before implementation. Owned folder only; T55/T52 sources remain unchanged.

## Method
For a column-ray simplex V, the inner triangle endpoints are P+r*v3, P+r*v4, P+r*v5. Construct an orthonormal tangent frame e1,e2 from v4-v3,v5-v3 once per cone. At each node project A=P+r*v3 into that frame, let delta be its perpendicular norm, and scale the fixed edge coordinates by r. The oriented local determinant is delta*r^2*lu*lv. Feed these coordinates into the unchanged degree-three spherical boundary-moment formula, retaining its resolution/cancellation/range guards and positive original-domain fallback. All endpoints retain the same signed-L1 plane and orthant.

Two alternatives, frozen before implementation:
1. Repeated tangent construction is not a material cost because trig/fallback dominates. Separate setup, analytic evaluation, fallback work and wall/CPU on identical endpoints; no claim from legacy loose 10000-operation allowance.
2. Fixed-frame reuse changes near-degenerate numerical behavior or creates spurious analytic acceptance. Compare all original d6 symmetric/asymmetric thin controls, zero r, permutations, and stored cone outer rules. Any discrepancy is retained and resolved by positive-domain reference before proceeding.

Initial frozen cycle: original T55 d6 fifteen width controls; first16 stored T78 cones in saved order at Jacobi outer orders4 and8; direct same-endpoint comparison to pinned T55 triangle_batch and independently positive quadrature for all controls. Observe maximum relative difference, fallback/guard counts and frame constructions. Do not change tolerance1e-11, min_tangent_ratio1e-5, roundoff_screen1e-10. Same-endpoint differences above1e-9 or changed fallback counts trigger diagnosis before a trained pilot. No new model inputs or MC. Time a repeated deterministic ordinary single-cone batch of4096 nodes; all setup charged. Follow with existing64-cone T52 pilot only after controls and independent review. No full93M-node integration in this phase.

## Status
Running: protocol frozen; implementation/control phase next.

## Results
None yet. Commands and source hashes will be recorded with each result.

## Compute accounting
Single-thread processes, <=2GB, review lock<=20CPU or compute lock<=110CPU per cycle. Count actual frame constructions, endpoints, analytic attempts, trig calls, fallback nodes and rule setup. Record full cold CPU/wall including fixed-frame setup separately from imports/build. A symbolic count will need a fresh line-group audit; inherited loose allowances do not establish algebraic speedup. No full portable estimator or sampling-win claim.

## Claims
[inferred] The exact tangent plane is constant when r>0, so one frame suffices algebraically.
[speculative] Removing repeated frame construction materially improves measured integration cost.

## Alternatives not ruled out
A compiled triangle kernel, additional degree-three algebraic simplifications, alternate ray order, different quadrature/planning, or full geometric acceleration. These are not part of the initial comparison. A failed finite control means implementation/numerical defect or inadequate frozen screen, not a proof against analytic integration.

## Failure modes
r=0 and collinear base edges require positive fallback. Floating endpoint construction and frame subtraction may disagree near degeneracy. Original topology, quadrature-bound arithmetic and cancellation screen are not directed certificates. The new implementation cannot strengthen those guarantees merely by matching finite controls. Source imports and counted fallback dependencies must be pinned.

### First control cycle
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock bash -c 'ulimit -v 2500000; ulimit -t 20; /workspace/clean-neurons/.venv/bin/python m16/approaches/fixed_t6_triangle/check.py'`

[measured] Fifteen thin controls, zero remainder and first16 stored cones at orders4/8 pass the frozen screen: max old/new relative difference4.93394e-11, no changed fallback counts. All fifteen thin/zero controls also use original-domain positive quadrature. Total.319345CPU, zero model forwards; raw diagnostics in control_results.json. This does not certify floating accuracy. Short per-case CPU clocks report zeros; the three4096-node repetitions do not establish a stable speed ratio. Source hashes retained.

Next frozen validation: all six endpoint permutations on the same fifteen controls and stored cone2/order8, positive-domain reference for the largest old/new difference in that fixed cone (diagnostic selection, not estimator fitting); then20 alternating old/new4096-node batches with full per-call frame/setup included, reporting wall and aggregate CPU. Independent T55 review requested. No tolerance or screen changes.

### Permutation and paired timing cycle
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock bash -c 'ulimit -v 2500000; ulimit -t 20; /workspace/clean-neurons/.venv/bin/python m16/approaches/fixed_t6_triangle/refine_controls.py'`

[measured] Six permutations of the fifteen thin controls and fixed stored cone2/order8 preserve old/new fallback counts. Maximum new permutation-relative difference9.812e-11. At diagnostic-selected node447, old/new/reference values are2.222033652230084/2.2220336522738324/2.2220336522643134; new differs4.284e-12 relative from positive quadrature. These ordinary arithmetic references are not a floating certificate. Twenty alternating4096-node calls (engine initialization, endpoint construction and new fixed frame included) total old/newCPU.180032/.131884 and wall.181967/.131078:1.365x CPU,1.388x wall in this ordinary case. Entire cycle.379819CPU. All raw observations retained in refined_controls.json; no favorable repetition selection.

COST.md separates the108*n−111 scalar frame-operation saving from any future tightening of the legacy10000-operation allowance. No estimator-win inference follows from this primitive speed.

The original T52 pilot caches are ignored and absent from the root checkout. Their owner has been asked for existing paths/hashes; no regeneration or alternate cone selection has been substituted.

### Frozen trained pilot (before evaluation)
After independent primitive review, reuse all64 indices `linspace(0,42011,64,dtype=int)` and exact orders from the existing T52 `plan_1e-06.npz`, not the favorable16-cone controls. Read-only original caches live under `/workspace/work/recover-small_h2_34/m16/approaches/small_ladder_review/portable_cost/t6/`; rays SHA25615ff556d1ef6168275d67a785eb7f2623a411cb8832816cf5cfc97060cd339c7, plan SHA2563126342f2705bf7db13483b603f423c0d9a036a02eb434b5628913850013ae81. Same inner allocation1e-8/42012, same counted positive fallback. Fresh engines/rule caches for each complete old/new pass; report full pilot CPU/wall and metadata, per-cone masses and all fallback-count differences. Stop/diagnose if old/new mass relative difference>1e-8 or fallback counts change; no screen adjustment. Save each cone before continuing. Separate original scalar cost from the frame-only difference and retain full partial mass/error (never label it full accuracy). The already completed original64-cone result is a consistency check, not fitted advice.
