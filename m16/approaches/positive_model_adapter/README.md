# T171 model-facing positive-mass audit

## Question
Can reviewed captured-corpus positive integration become a model-facing exact estimator with capture, N allocation and storage charged, without silently inheriting semantic defects? Claimed TASKS-only c3df8fef.

## Method
Pin existing estT source and inspect orthant→mass→cell-error→finish→estimator call flow. Orthant values feed only additive accumulators/returns; geometry/gate decisions are functions of model and cone state. Freeze synthetic sentinel-return parity and exact first-index tie controls before any trained capture or adapter implementation. Protocol records alternatives and source hashes.

## Status
Initial source/control audit in progress; no trained geometry or masses rerun.

## Results
Pending. Analytic active-identical counterexample is H2,Wi=(1,-1),Wh=0,Wo allones: all four logits equal |x3| almost surely, true first-index accuracy1/4. Original sequential zero-margin clipping appears to return accuracy1.

## Compute accounting
One thread,2GB,<=20CPU review lock. Synthetic callback returns are arbitrary sentinels, not Gaussian masses or estimator values. Import/cold setup costs retained as CPU but not claimed complete native arithmetic. Future two-pass collection needs explicit ray copies, storage, N selection and summation accounting alongside enumerator and exact/library costs.

## Claims
[inferred] Purely additive source call flow permits delayed evaluation without geometry feedback. [inferred] Positive-volume identical output ties need first-index ownership. Neither claim is a full floating topology/cost certificate.

## Alternatives not ruled out
Trained artifacts have no recorded identical active row pairs; generic tie failure does not refute their conditional mass results. Nonidentical proportional/face-restricted ties, geometry roundoff, source tolerances and exact cost remain separate.

## Failure modes
No generic adapter adoption before tie controls and separate independent review. Source original remains unchanged; shared fixes DRI-only.

Initial audit complete in0.636CPU (`audit.py`, `audit.json`): all9 sentinel cases preserve the ordered ray stream and source STATS. Alloff model captures72 identical rays for each sentinel; two active-identical controls each capture zero error cones and return1, although exact first-index accuracy is1/4. This is an inherited semantic failure independent of Gaussian integration, preserved unchanged. AST locates nine callsites; manual source inspection shows each mass return is accumulated or returned and never used in geometry branches. The claim assumes finite successful callbacks: exceptions, deadlines and budget guards can change termination.

Reproduce only when needed: one-thread2GB review-lock `/workspace/clean-neurons/.venv/bin/python m16/approaches/positive_model_adapter/audit.py`. No network inputs, trained geometry or Gaussian integration were evaluated. A local exact-zero output-margin ownership repair is the next bounded proposal, requiring independent source/control gate; shared source remains DRI-owned.

Local guard and model-facing prototype: `guard.py` handles exactly zero stored coefficient margins with first-index ownership, preserves original arithmetic for nonzero margins, and separately counts four scalar equalities, three Boolean reductions per checked T4 margin, plus orientation comparisons only on ties. `adapter.py` collects copied4x4 rays from the actual model, then assigns1e-7/N using the actual count, conservatively rounding down at most one ulp if needed. Numerical selectors/kernels remain pinned and unchanged. Exact-model vs floating-coefficient ties remain distinct; this is not an outward neural-geometry certificate.

Frozen seven controls (`controls.py`, `controls.json`) ran2.617731CPU. Six active constant-label models returned1/4 within1.21e-12, with conditional bounds1.50–1.56e-8 and216 cones each. They include zero/nonzero identical outputs, ties whose first winners are0/1/2, and strict winner3. The alloff model captured72 cones but the first failed the finiteorder32 target: exactKKT bound3.633462735e-8 versus1.388888889e-9. No estimate was returned for that case, no cap/target changed, and no Gaussian rule was evaluated on the rejected first cone. The run-complete flag means all seven controls were attempted, not all returned. This is a finite selector limitation, not a tie-guard failure or general positive integration impossibility. Exact structural constant-output recognition or geometric subdivision remain plausible distinct rescues.

Import review found a separate implementation issue: original pinned `m43.py` reads unrelated `m43_weights.npz` at import despite only Ops being used. `adapter_initial.py`, `import_failure.json` and original controls preserve that version. Current loader extracts only the pinned Ops class AST and removes only its import from engine AST; numerical bodies stay unchanged. `check_loader.py` forbids np.load and compares actual bytecode for14 engine functions plus3 Cone methods against original-source execution with an Ops-only import stub. All pass; no integrations rerun. Initial source-text-only inspection was strengthened to actual bytecode comparison, with original check artifacts retained.

The current prototype is not a completed arc_eval estimator: geometry/source proxies, cold chart generation, Fraction/MP and native kernels, metadata validation/import, capture/list/header/copy/storage and final fsum costs are separately exposed or unresolved. It also returns unresolved on finite selector failures. No sampling comparison or complete hardware cost is claimed.

T55 initial guarded control/import review passes a558c8a9. Capture-only trained validation now complete: source checkpoints0/1 produce all35098/30726 rays bitwise identical to prior T131/T148 arrays, identical STATS and zero tie-ownership branches. Actual captureCPU10.775686/10.642498; copied4492544/3932928 bytes, plus unknown array/list headers. New scalar equality+Boolean work164409/159194. Inherited geometry proxies35650854/33043583 exclude cold/native parts and are not portable costs. Prior positive results can be composed because every ray matches, with all142873286/102122861 prior mass units and exact/library terms charged; replayed saved values are not fresh estimates or free advice.

Initial seed0 validation hit SIGXCPU110 because lazyNPZ reference indexing inside each comparison repeatedly decompressed the whole archive. `capture_parity_initial.py` and `capture_initial_failure.json` preserve source/exit and110CPU budget charge (exact timing unavailable). No numerical failure inferred. Current validation loads reference once and checkpoints complete capture before comparison; no masses rerun. `captured_seed*.npz/json` allow subsequent review without geometry repetition. Successful process CPU11.768478/11.746696, plus failed110CPU budget separately.

Seed0 known-category composition (`cost_composition.json`): source visible geometry42396504 + coldhalves30288 + positive mass142873286 + guard164409 + still-executed unused original quadrature-import542 =185465029. Source h4 census separately confirms542=9*(12+48)+2; doubled candidate rule setup is absent. Additional48 native inverse numeric envelopes13920 give185478949 in this enlarged subset. Rank/SVD275897 calls, exact/library/native controls, capture/storage and allocation/finalfsum costs remain separate. Three additional named allocation/final-return operations are listed outside the subset. No seed0 geometry count is transferred toseed1, and no full-cost/sampling claim follows.

T55 final saved two-seed capture and cost audit882efc72 passes all array/source/hash/STATS, guard/copy and composed-cost identities. No integrations rerun; generic alloff failure and110CPU failed diagnostic remain explicit.
