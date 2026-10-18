# T267 H4T3 closed-angle comparison

## Question
Does exact internal-facet cancellation save Gaussian integration work when each
three-dimensional cone already has a cheap closed solid-angle formula?

## Method
The fixed existing H4T3 weight archive is decoded exactly into dyadic weights.
The unchanged rational geometry engine captures gates and correct cones once.
The reviewed boundary-chain reducer cancels identical oriented facets and cones
the remainder to fixed orthant anchors. `angle.py` evaluates original and signed
reduced triangles using the positive-denominator trihedral atan2 formula.
`PROTOCOL.md` freezes six analytic controls and all provenance/resource limits.

## Status
Sole producer run complete, raw `b1337926`, published `6d7f45c1`.
Source/resource review `8c0b1de9` was published `c3109eac`; immutable execution
gate `29acef0e` was published `887a06bc` before execution. Independent saved audit `e18bedc1` by recover_h4, assigned in `40a8caf2`, passes
all 70 gates, 86 correct cones, 176 gate and 244 correct simplices, exact readout/
rank/incidence/chain/scalar records and 1,536 progress events. Review CPU was
0.25197 seconds with 64,436 KiB peak RSS; no author or atan2 calls. No failed producer
attempt or completed rerun occurred.

## Results
All six analytic controls passed. One capture produced 70 terminal gate cells,
176 gate simplices, 86 correct cones and 244 correct simplices. The exact total
normalized gate determinant is 8. This is an aggregate conservation check, not an
independent proof of partition completeness or disjointness.

Of 496 canonical facet groups, 236 cancel; 260 remain, including 60 zero-anchor
cones. There are 200 integration terms: 149 positive and 51 negative. Thus the
fixed integration count falls 18.03%, from 244 to 200. The determinant absolute
weight increases about 1.417%; the exact dyadic enclosures remain in the raw file.

Baseline Gaussian value is 0.9845503813585571; the signed boundary value is
0.9845503813585572. Their 1.11e-16 difference is a finite floating diagnostic,
not a directed floating error certificate. No quadrature is used in this rung.

## Compute accounting
Command, run once after published gate:
`bash review/facet_cancellation/h4t3/launch.sh`

The process used 0.96519 CPU seconds and 64,768 KiB peak RSS, with recorded
20/20 CPU and 2 GiB limits. New decode/capture/classification/triangulation and
recording took 0.60039 CPU seconds. Boundary construction/recording took 0.03704.
Baseline integration/recording took 0.01107; boundary took 0.00729, a saving of
0.00377 CPU seconds. Timers describe this instrumented run, not stable throughput.

Each angle call has 45 named arithmetic categories and nine rational coordinate
conversions, excluding conversion internals, libm implementation and control/I/O.
Baseline has 10,980 categories and 2,196 conversions; boundary has 9,000 and 1,800.
The six controls add 270 categories and 54 conversions. Geometry and boundary
integer work are separately retained; they are not interchangeable with FP counts.
Boundary work includes 740 exact determinants, 13,320 integer multiplications,
4,440 integer accumulations, 2,196 GCDs and ray divisions, and 732 facet lookups,
coefficient additions and identity terms. All 1,536 progress events are retained.

## Claims
[measured, independently saved-audited e18bedc1] Identical-facet cancellation reduces
the fixed closed-angle call count, but its measured construction/recording cost
exceeds the saved integration time. This run does not demonstrate a cold speedup.
[inferred, source-reviewed] Exact chain orientation preserves integration under
the stated inherited-complex and anchor premises.

## Alternatives not ruled out
Different anchors, compatible subdivision merging, cross-orthant matching,
amortized repeated integrands or direct boundary integration may behave differently.
This small fixed model does not establish large-width scaling or an ARC estimator.

## Failure modes
Inherited geometry completeness/disjointness and checkpoint labeling are qualified;
the archive bytes are authoritative and no fresh checkpoint equivalence is claimed.
Signed cancellation, conversion/libm error and setup overhead remain relevant.
No sampled inputs, RNG, training, checkpoint download or H4T4 replay occurred.
