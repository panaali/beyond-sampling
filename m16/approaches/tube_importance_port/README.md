# T20 cold per-model importance weighting port

## Question
Can the remaining fitted-position importance variant be made reproducible with all model-dependent setup charged, while retaining the corrected interval-native radial walker? Claim699cf3d was pushed before implementation. This is a port/cost audit, not a new accuracy or sampling-win claim.

## Method and frozen protocol
Read-only tube source `tube/tube.py` SHA09fc08e0e3386b664971f4cbf003824c2ff44b11fe85f7cda1b7f8f7cc60e718 and estimator SHAe1a9ec49e5370bbc2dde35da320d4c866b9b6c36321b2919b89d019c5bc9254f. Reuse the reviewed `feynman_incremental.radial.walk_mass` and model-free face generator; do not edit either. Public interface est(model,budget,rng) returns estimate and actual charged numerical proxy, with richer ledger available.

Original q23/q12 are model-dependent advice, learned on the original model. README reports16,384 pilot faces per family and roughly280,000 old-style forward equivalents; the exact table-fitting script/cost ledger was not found in this folder. Pin table hashes fb01ea23e82324990c7e94ff3f354b406bc463a42f13801e94ca2d0795c7c4c6 and6ae92ae1e7fc5709c8f5cd13233872ded07a570657bad079320bd4ce5ee1b5c9. An advice-mode diagnostic may reproduce them, but its total preprocessing cost remains unknown and must not be labeled complete.

Cold production variant: deterministic pilot2,048 R23 faces and256 R12 faces on each supplied model, uniform positions, shift=(original SHIFT+sqrt(2)−1) mod1. Group actual combo IDs, compute sample standard deviation of integrated face mass divided by sqrt(mean charged per-row walker proxy), mix80% normalized scores with20% uniform. If every score vanishes, use uniform. Fail explicitly if any stratum has fewer than2 pilot points. Pilot values are not reused as production contributions. Production uses original fixed SHIFT and original budget/face counts. This is deterministic quadrature/adaptation; no randomized-unbiasedness assertion.

Frozen comparisons: original public checkpoint budget512 and2048, plus all-zero readout at512; three modes uniform/advice/cold, no new Monte Carlo or noisy-model sweep. Existing exact zero-label per-face formulas can test weighted aggregation; overall constant-label truth.1 does not remove outer quadrature error. Budget is advisory; fixed pilot/setup may exceed it and actual cost must be returned.

## Falsifiers and alternatives
1. Importance correction or grouping may be wrong: independently enumerate finite strata and verify sum q_j*f_j/(K*q_j)=mean(f), positivity/floor, permutation invariance, constant/all-zero score fallback, and actual-pilot combo coverage. Hash nodes/faces/proposals and preserve all failures.
2. Gains may come from advice or free pilot cost: charge every pilot face, radial walk, table initialization, transform, fit and production weighted reduction. Compare cold to uniform and advice only at frozen nodes/budgets; advice fitting cost remains explicitly unresolved. Do not adopt historical MSE claims from legacy proxies.
3. Corrected walker and old source can disagree on positive-volume ties: zero-readout control uses first-index predictions and independent per-face radial masses, not legacy's zero-slope tolerance rule.

## Compute accounting
One thread/2GB; batches<=65,536, cycles<=110CPU; review lock only for<=20CPU cycles. Complete cold model-dependent work and numeric setup charged, but inherited table/Sobol/beta/CDF proxies and interpreter/import exclusions remain nonportable. Return partial costs on failed walks; invalid input is distinct from implementation/budget failure. No opaque kernel is called free. No full MSE ratio without an independently accepted operation convention.

## Status and claims
Frozen before implementation. [inferred] Conditional importance correction is algebraically valid for a fixed positive proposal. [speculative] Per-model pilot weights may improve outer quadrature enough to offset their cost. No measurement or success claim yet.

## Alternatives not ruled out and failure modes
Small pilots may learn poor weights; deterministic pilot/production coupling can matter. Native walker epsilon/root behavior, radial beta cancellation, fixed outer quadrature and blind-region error remain inherited limitations. Newton and adjacent-component variants are separate unported methods. Original table-fitting cost/provenance is incomplete; a failed cold variant would not retire importance sampling or deterministic importance quadrature generally.

## Bounded matched comparison (2026-10-17)
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python -m m16.approaches.tube_importance_port.check_port`
completed all nine frozen combinations. Original checkpoint estimates at budget512: uniform .949256087952, original advice .956874663052, cold .961631595385; at2048: .951360624975, .954311913238, .955103287242. These are estimator outputs, **not an accuracy/MSE validation**. Cold known numerical proxy costs49914.997/51976.279 forward equivalents include all2304 pilot faces, compared with uniform19721.754/22181.848. Advice costs19554.302/21819.206 exclude its unknown historical fitting work and are incomplete. Cold fixed setup can exceed the advisory requested budget.

The all-zero readout control has exact accuracy.1. At512, uniform .044954994237, advice .077187168193, cold .098304631600. Preserve all three finite outer-rule biases; the closer cold result is one control, not robust superiority. Every uniform output exactly matches the previously tested corrected uniform port. Proposal identity, permutation invariance, zero/constant-score fallback, invalid pilots/budgets and an injected walker failure pass, including its321 known operations in the partial ledger. Actual per-stratum pilot coverage is recorded, not assumed.

### Frozen independent weighted-face check
Next reconstruct each zero-readout production face's exact error mass using positive scalar quadrature of A(1+t²)^−5 and first-index target logic, without network recurrence, native interval integration or incomplete-beta calls. Rebuild uniform/advice/cold faces from the stored proposals, compare both face and mass fingerprints where applicable and each weighted aggregate. This distinguishes a wrong radial/importance calculation from outer quadrature bias. No extra models or node sweep.

The frozen independent positive-quadrature check passes: `python -m m16.approaches.tube_importance_port.check_constant` reproduces the three weighted zero-readout estimates within2.26e-13, with exact production face hashes and no network/beta calls. CPU.034111 seconds. Its reported quadrature errors (~1e-14) are not rigorous bounds; the2e-13 native difference is retained rather than equated to the reference's own integration estimate. The sizable uniform/advice and smaller cold discrepancies from.1 therefore arise in the finite outer rule, not the tested radial weighting. No sweep was used to select the cold result.

Status: bounded port and controls complete; independent agent review requested, DRI classification pending. The choice std/sqrt(cost) is a within-stratum allocation heuristic, not a proved optimal proposal for the implemented mixture estimator. For iid mixture importance with fixed counts not stratified, the second moment E[f²|j], rather than Var(f|j), controls the objective; deterministic nodes create further differences. This alternative remains open and is not silently substituted after seeing outputs. The20% floor preserves support. Measured gains here do not establish general efficiency, robust MSE or a solution of the challenge.
