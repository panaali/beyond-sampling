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

## First fixed-control checkpoint (88ff3af7)
[measured, independent saved audit pending] All10 fixed fixtures passed in
0.003997934CPU. The original narrow3 returns a positive rational upper bound
5764607523034234881/23926103924128485737632800186396835844, outward binary64
2.409338161078899e-19. Zero-axis and four fallback fixtures return1 with explicit
reasons. Orthant bounds are conservative but loose: approximately.35355,.33333,
.43301,.48 for dimensions2..5, versus exact cone masses1/4,1/8,1/16,1/32.
No trained utility or speed inferred. Maximum recorded fraction size125bits;
ledger counts are macro diagnostics only. Original cap is not rerun.

Process deviation: immutable gate1ed5296b was committed before execution, but
its remote publication follows the run, contrary to the reviewer's requested
published-gate order. Sources and independent theorem/source gate were already
published5c5f3896/d84f5236. Raw hashes and chronology are preserved; this is not
silently described as full publication-order compliance and no rerun is used
to rewrite that history. Independent saved review will distinguish numerical
validity from this administrative ordering defect.

Reproduction in fresh exclusive output paths: `python
review/conservative_cap_bound/controls.py`, with gate.json and external20CPU,
2GB,one-thread review lock. Existing results refuse overwrite.
