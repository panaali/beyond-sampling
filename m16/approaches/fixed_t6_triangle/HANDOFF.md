# T104 bounded handoff

Claim a1d6a50; research4aca5a5,e95a4a6,e3539a8,c1e93c2,610789e,bcf0b12. Local files only, no shared modifications.

The exact frame identity is valid: for A=P+r*v3, B=A+r*(v4-v3), C=A+r*(v5-v3), positive r preserves the fixed tangent basis and scales both tangent lengths. The original triangle moment, cancellation/range screens and positive fallback are preserved. Degenerate r=0 remains a fallback.

Frozen finite validation covers fifteen thin controls and zero remainder, first16 stored cones at orders4/8, six permutations, and the original64-cone T52 pilot/orders. The pilot uses167352 inner triangles and identical3641 positive fallbacks. Partial masses agree within4.34e-19; max per-cone relative difference7.45e-12. Full numerical old/newCPU.499089/.437942, with artifact writing separately. Ordinary4096-node benchmark20paired calls gives1.365xCPU. These are finite observed runtime gains, not an end-to-end accuracy or portable-cost victory.

True frame arithmetic saving108*n−111 is separated from the legacy10000-operations-per-triangle allowance. Per-cone validation bookkeeping remains part of the inherited setup allowance; these differences are not a new full audit. T52 floating geometry, screening and quadrature-bound arithmetic qualifications remain unchanged.

T55 independently reviewed the identity and reproduced sensitive triangle values with GL64/128; one-row versus512-row dot reductions exposed a floating cancellation sensitivity. The frozen rowwise-einsum rescue makes all512 scalar/batch values identical, but can worsen numerical error at the selected node. Both variants remain qualified; the original primitive remains the benchmark default. Independent review commit will be appended when available; the review message is already delivered to root/author.

Next plausible step is a separate p3-only edge formula/counting task, removing unused solid-angle and half-angle trig calculations. Do not launch a93M-triangle full integral merely from this modest pilot. T6 native geometry and broader triangle representation alternatives remain open; no method family is retired.
