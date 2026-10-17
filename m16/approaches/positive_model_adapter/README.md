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
