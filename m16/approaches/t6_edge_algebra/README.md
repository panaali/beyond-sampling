# T106: degree-three spherical edge algebra

## Question
Can the six-input triangle primitive remove redundant trigonometry while preserving guarded accuracy and reducing actual and honestly counted cost? Claim19e734f published before implementation; T104 independently reviewed59fae3f.

## Method
For normalized edge rays u,v, let c=u·v, s=||u×v||, theta=atan2(s,c). The midpoint-frame coordinate squares are aa²=(u0+v0)²/[2(1+c)] and bb²=(v0−u0)²*(1+c)/(2s²). Thus the required squared-coordinate edge integral is aa²*(theta+s)/2+bb²*(theta−s)/2. Keep the existing small-angle positive Taylor difference and explicit truncation bound. Remove the unused solid-angle calculation,9 sine and3 cosine calls per triangle. Coincident edges fall back without evaluating a zero denominator. The original fixed-frame matrix-vector projection remains unchanged; T104 rowwise alternative is not silently substituted.

Freeze TWO alternatives before results: (A) use raw edge s,c, justified by the exact unit-ray identity; (B) normalize s,c by sqrt(s²+c²) before the half-angle identities, preserving theta. Both are reported regardless outcome; neither is selected by reference accuracy. T55 independently checked the algebra; numerical behavior and cost remain to test.

Alternative explanations and falsifiers:
1. Trig removal saves nominal operations but extra arrays/divisions worsen CPU. Measure alternating4096-node calls with full engine/frame/setup, plus unchanged T52 64-cone pilot if controls pass. Separate raw/normalized, no favorable repetition selection.
2. Ray normalization error or small-angle cancellation changes fallback acceptance and produces erroneous mass. Reuse all15 T55 thin controls, r0, first16 stored T78 cones GL4/8, six permutations of thin controls/fixed cone2; original-domain positive references for sensitive cases. Preserve every failure and inspect changed fallback sets before proceeding. Thresholds stay1e-11 inner tolerance,1e-5 tangent resolution,1e-10 roundoff screen; initial old/new relative diagnostic gate1e-9. No formal roundoff guarantee.

Protocol: both variants run the original T104 control suite with new filenames; explicit s0/coincident controls and fixed sensitive node447 record values and positive reference. Then20 alternating ordinary4096-node calls per variant versus original T104. Only after independent source/control review use same64 T52 cones/orders/inner allocations with one cold pass per variant. No full93M-node evaluation. A fresh line-group cost inventory must charge the original T104 baseline AND each new path, distinguish true removed calls from tighter legacy allowances, and include frame/rule/fallback/setup work.

## Status
Running: frozen protocol, implementation next. Owned t6_edge_algebra only; no T55/T52/T104 source modification.

## Results
None yet.

## Compute accounting
Review lock<=20CPU or compute lock<=110CPU, one thread,2GB RSS cap, chunks<=65536. No model forwards, random inputs, fitted constants or new MC. Record source hashes, every tested variant and artifact I/O separately. No estimator cost activation until independent audit; the947B historical T52 allowance is not evidence of equivalent new algorithmic savings.

## Claims
[inferred] Half-angle squares follow exactly from midpoint coordinates for unit rays with0<theta<pi. Same-orthant endpoints keep1+c away from zero.
[speculative] Reducing trig calls improves measured complete primitive/pilot cost while preserving finite screened accuracy.

## Alternatives not ruled out
Compiled evaluation, directed bounds, adaptive precision or more stable boundary-moment summation; these are distinct follow-ups. No broad impossibility follows from a failed variant.

## Failure modes
Near-coincident edges and cancellation need positive fallback. Raw s,c need not satisfy the unit circle identity exactly in binary64; normalized pairs follow a different floating path. Existing heuristic screen may miss small absolute errors. Conditional outer integration bounds do not certify gate geometry or floating boundary moments.
