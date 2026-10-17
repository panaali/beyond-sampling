# T106 bounded handoff

Claim19e734f. Protocol3cc4a2f, source/controls/costb98bd20; final pilot recorded in this commit. Independent T55 source/circuit auditcaae08f.

Two frozen variants remain: raw edge sine/cosine and pair-normalized edges. Exact midpoint-square formulas remove an unused solid angle and12 half-angle/sine/cosine evaluations. Small-angle remainder and positive fallback preserved. No shared sources changed.

Thin/coincident/permutation/stored-cone controls pass at maxrelative4.291e-13 from T104 with no changed fallback counts. Original64-cone T52 pilot sums identical.003088193770479861; per-cone raw/normalized relative1.39e-14/3.35e-14. All3641 fallbacks unchanged. CPU improvement on this finite pilot is only2–3%; ordinary4096-node benchmark shows a larger but still modest difference. Do not infer a large speedup from accounting refinements.

Fresh independently audited arithmetic-circuit ceilings: baseline/raw/normalized1648/1360/1456 per triangle plus512 frame and2048 batch, unchanged counted fallback and rules. Pilot313.19M/264.99M/281.06M operations. All are partial-integral costs; original geometry/planning and floating numerical caveats remain. DRI must accept any full estimator activation separately.

Next falsifiable opportunity: batching existing positive fallbacks by unchanged selected quadrature order (all pilot fallbacks order4) with bounded node chunks and retained scalar reference. No implementation under this claim; register/claim separately. Full integrals remain deferred.
