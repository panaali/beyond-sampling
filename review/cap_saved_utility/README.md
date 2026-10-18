# T209 conservative cap utility on saved exact-reference cones

## Question
Are the newly safe cap bound and a saved-certificate determinant bound tight
enough to merit a later pruning-adoption question on this fixed diagnostic set?

## Method
Reuse64 original T193 pilot rays and exact KKT certificates, unchanged T200
cap source, and saved numerical references. THEORY.md derives the determinant
bound; PROTOCOL.md freezes scope and per-cone thresholds. No integration or
geometry replay; no kernel/cost-chain continuation.

Two competing explanations are deliberately separated: cap width may ignore
thin sliver geometry, while even a tighter certificate-based bound may be
expensive to construct. The fixed same-ray comparison diagnoses the former;
existing construction ledgers remain visible for the latter. No production
cost conclusion is authorized by this diagnostic.

## Status
Claim2288e3a9 pushed; source/provenance and theorem being frozen. No new cap
calls or saved-bound contraction yet. Independent gate required before run.

## Results
Pending fixed64 bounded phase. Conditional numerical references are not full
floating/geometry certificates. Every loose bound, fallback and mismatch will
be retained, without selecting another subset or axis.

## Compute accounting
Source/theory work only so far. Planned64 cap calls,64 exact saved contractions,
zero model/geometry/KKT-solve/mass/reference calls, <=20CPU/2GB review lock.
Certificate construction is inherited evidence, not free production work.

## Claims
[inferred, review pending] Both bounds enclose represented cone mass under the
stated exact-ray/certificate assumptions. [speculative] One may be useful for
pruning; practical tightness and implementation overhead remain unresolved.

## Alternatives not ruled out
Other axes, exact spherical bounds, cheap outward floating implementations,
other simplex bounds, lower-dimensional or class-restricted pruning. No finite
64-cone diagnostic proves general failure or coverage.

## Failure modes
Changing ray normalization without updating certificates; treating conditional
references as exact; converting a bound downward; hiding fallback1; counting
already-built exact certificates as free; confusing cone mass with total
classification error or using per-cone thresholds as a global budget.
