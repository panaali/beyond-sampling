# Fresh triangle line-group ceilings

This is a proposed conservative arithmetic/comparison model, pending independent audit. It replaces neither the historical T52 record nor DRI-approved estimator costs. Both the unchanged T104 baseline and the new variants receive fresh ceilings, so most reduction from the old10000-per-triangle allowance is explicitly an accounting refinement, not a new algorithm.

Convention: scalar add/subtract/multiply/divide, abs, comparison and Boolean combination each1; sqrt and each transcendental20; fixed integer powers use at most(k−1) multiplications plus one reciprocal for negative powers. Float exponent values that are exact small integers follow this convention (as earlier T71). Array allocation, access, vector-library dispatch and Python execution are not literal FLOPs; record actual CPU/wall separately. Setup/header overhead charged2048 per batch and frame validation/construction512 per cone. Positive fallback/rule setup use the unchanged independently reviewed T52 counted engine. Generic shape assertions and exponent constants are within batch setup. No uncharged library SVD/QR/solve occurs here.

| Group, per attempted triangle (charge all input triangles) | T104 baseline | Raw edges | Normalized edges |
|---|---:|---:|---:|
| Reconstruct18 endpoint coordinates, finite/sign/L1 and shape-dependent validation | 192 | 192 | 192 |
| Two6D projections, perpendicular norm, scaled edges, endpoint norms, resolution/local coordinates | 208 | 208 | 208 |
| Three3D ray norms, nine normalizations, three cross products/norms/dots/atan2, boundary normals and determinant | 320 | 320 | 320 |
| First moment and absolute first-moment terms | 32 | 32 | 32 |
| Unused solid angle (products, sum, ratio, atan2) | 64 | 0 | 0 |
| Degree-three edge integral, small-angle polynomial, boundary sum and explicit series remainder | 640 | 416 | 512 |
| Cancellation/range screens, final scaling, finite tests and per-triangle statistics reductions | 192 | 192 | 192 |
| Total | 1648 | 1360 | 1456 |

Counts deliberately cover attempted values even for triangles sent directly to positive fallback, then add all fallback work separately. Common-boundary group320 includes safe-sine predicate/selection overhead. Raw edge group416 covers aa²/bb², comparisons and the worst case that all three edges need the degree11 Taylor expression and degree13 remainder. Normalized512 adds three square roots, six divisions, circle squares/sums and validity predicates. A6D norm uses11 arithmetic+sqrt20; a3D norm uses5+20. Original degree-three group includes9 sin+3 cos calls (=240) and all remaining power/polynomial/sum work. The first-moment terms and repeated absolute-boundary products are counted separately.

Actual removed calls, independent of these ceilings: per analytic triangle, unused solid-angle atan2(1), sine(9), cosine(3). Raw replaces half-angle trigonometry by positive rational expressions in existing s,c; normalized adds3 square roots and normalization arithmetic. The group-ceiling difference288/192 per triangle includes different conservative slack and must NOT be represented as an exact hardware-operation saving. T104's frame-only arithmetic difference is documented separately; no historical count is silently rewritten.

These are finite-dimension d=6 bounds with128-node positive fallback cap, not asymptotic constants fitted to measured accuracy. Cold rule/frame/fallback setup and deterministic planning/geometry remain necessary in any full estimator. The current pilot is only a partial integral; no portable-cost activation or equal-cost MSE claim follows yet.
