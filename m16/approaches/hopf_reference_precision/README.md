# Fixed H4T4 reference precision

## Question
Can a separately fixed, more precise evaluation reference resolve T183's
inconclusive existing10M truth intervals on the unchanged25-model panel?

## Method
Claim c78c0999 precedes implementation. Protocol pins the25 existing models,
original Hopf estimates/costs and zoo44 forward. Independent named CRC32 seeds
per model, float64 Gaussian inputs,100M draws each in100 fixed1M blocks.
Return only integer hits from the otherwise unchanged original mc_acc body.
Original first-index predicted argmax and argsort second-rank labels retained.
First seed0clean block0 is reused once, after deterministic forward controls;
expansion requires independent source/pilot review and projected reference
work<=600CPU. Never stop or choose sample size based on accuracy outcomes.

## Status
Protocol/source preparation; no reference input draws yet. Only controls and
first block are enabled by the entrypoint. No estimator recipe change.

## Results
Pending. Prior T183 all60 point ratios<1 but75-family truth boxes inconclusive.
No matching reviewed positive references exist beyond clean0 (owner34 checked).

## Compute accounting
All reference draws are diagnostic evaluation, not estimator input work.
Record exact integer hits, RNG adjacency, per-block CPU and source/model hashes.
Extra deterministic controls are separately counted. Each process one thread,
2GB,<=110CPU via external limits/shared primary lock; tiny controls review lane
<=20CPU. First1M counts toward100M, never discarded. Original estimator costs
are frozen from T183; truth sampling cannot influence the deductive estimator.

## Claims
[inferred] Fixed100M truth should narrow sampling uncertainty by sqrt10 versus
old10M; this does not imply any particular performance conclusion.
[measured] T183's cost/source/model audit and point estimates are complete.

## Alternatives not ruled out
The observed estimator error may exceed sampling after precision improves;
keep that result. Full native cost/root-mask issues remain even if statistics
favor the estimator. Positive geometric references could eventually replace
sampling uncertainty but are not available for these24 other models.

## Failure modes
State discontinuity, repeated first block, shared seeds, changed forward/dtype,
accuracy-dependent stopping, dropping failures or treating floating network
truth as an exact-real certificate. Resource failure is not estimator failure.
