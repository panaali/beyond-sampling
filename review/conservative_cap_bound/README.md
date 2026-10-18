# T200 conservative represented-cone cap

## Question
Can a bounded local primitive return a conservative cap upper bound on exact
stored binary rays, including the T198 narrow-cone and undefined-axis controls?

## Method
Exact rational axis/cosine calculation; elementary dimension2..5 upper bounds,
integer upper sqrt and explicitly checked outward binary64 conversion. Fixed
resource/domain fallbacks return1. THEORY.md and PROTOCOL.md freeze scope.

## Status
Claim804f5456 pushed. Theory/source being frozen; no controls executed.
Independent reviewer78 assigned. No estimator adoption or trained replay.

## Results
Pending10 fixed analytic/domain controls. T198 historical result is reused as
a pinned artifact; the defective cap is not rerun.

## Compute accounting
Source/theory work only. Future single20CPU/2GB/one-thread control process.
Expression/bit inventories are diagnostic, not full native or runtime costs.

## Claims
[inferred, review pending] Exact rational containment plus conservative scalar
bounds can avoid T198's zero-underestimate. [speculative] A practical floating
implementation or pruning use may follow; no such efficiency claim here.

## Alternatives not ruled out
Rational overhead may be too high; elementary bounds may be too loose; direct
outward floating norm/dot bounds or other axes might be better. Domain coverage,
retained recipe error and numerical mass contributions remain separate.

## Failure modes
Mistaking represented-hull containment for full estimator correctness; missing
subnormal conversion; masking resource fallback as a tight bound; claiming
expression counts equal runtime or extrapolating analytic controls to models.
