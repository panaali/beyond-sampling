# T119 independent T109 cone-mass review handoff

Author snapshot: commit7b7279a, SHA6617ddd0a042d4af5fbc899e0542b9e31c1ee23f5ab8d47e5fb8ee2c005f05a6. Source is unchanged. Claim e4c9c77 preceded implementation. Tests use captured T110 exact geometry, not re-enumerated or re-selected trained cones.

## Findings
- All72 independent analytic product/interface/scaling/permutation checks and3 skinny positive controls pass1e-7 absolute accuracy.
- Frozen quadratic6347-cone corpus gives .477013704054524 vs independent .477013125994119, but cone5826 is negative−3.6109944e-7.
- Frozen nearest11421-cone corpus has a domain exception at420 and negative495/10506; its unmodified aggregate is not reported.
- Independent positive or50/70-digit Schur references verify all four failures. True masses5826/10506 are2.66355056e-10,420 is3.32844204e-8,495 is8.46082735e-10. All16 originally frozen reference checks agree with source within9.54e-10. This distinguishes localized implementation failures from a family-wide negative.
- Increasing source quadrature to160 nodes does not rescue the major negative masses on these inputs. Nearest420 is asin(−1.0000000000000002) after recursive flat decomposition. Clipping that domain alone would not establish accurate masses. Signed cancellation and endpoint conditioning remain plausible causes, not proved diagnoses.
- A frozen review-only fallback replaces negative/exceptional source calls by positive GL20 when original rays share an orthant. A fixed48-candidate rational rotation catalog extends this to the remaining mixed495. Candidate9, pair(0,2),c255/257,s32/257 gives8.46082735079679e-10, agreeing with prior MP reference to4.24e-24. The rotation is exactly orthogonal in rational arithmetic; double geometry errors are outside the9.73e-17 integration-only bound.
- With four scoped replacements, no negative/failure remains on either captured corpus. Quadratic estimate .477013342688728 (+2.17e-7), nearest .477013161657630 (+3.57e-8). These are bounded diagnostic improvements, not global accuracy certificates for source-positive cones, a repaired shared implementation, or a solved rung.

## Cost and review qualifications
Author raw counters are3,288,370 and5,746,100 across the two corpora, .260/.531CPU cold diagnostic times. The local same-orthant wrapper replays at .246/.404CPU; rotation and saved-array update are .0135CPU total, mixed fallback .004CPU. Short clock readings quantize; no reliable microbenchmark speed claim follows. Each fallback has312,000 known integrand arithmetic operations plus positive-rule/node/determinant setup; rotations cost24 arithmetic operations per candidate plus coefficient/sign work. These are partial conventions and all failed original attempts remain charged.

In particular, `_gl` constructs Legendre rules without any author counter additions; cold order lists and CPU are preserved. `corr_from_normals` executes `U @ U.T` (a full4x4 product) while adding6*7 for unique offdiagonal dots: under the nominal four-mult/three-add per dot convention, the expression has16*7, a difference70 per call unless a symmetry-aware implementation is actually substituted. `np.linalg.inv` is charged64 as a proxy, not a verified elementary-operation ledger. This review does not certify a portable total or claim the <300 average-op goal: quadratic author count alone averages518.10.

T55 independently reviewed all-correlation Schur derivation, orientation of facet normals, positive orthant reflection, integration-only bound qualifications and signed-decomposition hemisphere validity. Final finite-rotation check was also sent for review. Ordinary float ray geometry and independent aggregate reference error remain limitations.

## Reproduce and next actions
Run modules `check_controls`, `replay --kind quadratic`, `replay --kind nearest`, `localize`, `independent_reference`, `check_guard`, `check_rotation` under one-thread/review or primary locks and the recorded CPU/memory caps. Saved results and original source failures are complete; there is no reason to rerun all experiments just to inspect them. See script CLI before replay/reference runs.

DRI/T109 owner should choose stable endpoint/cancellation repairs or a bounded positive representation fallback, test captured failures, and audit real setup costs. A source-level clipping guard alone is insufficient. Higher quadrature order alone is a tested unsuccessful rescue on these finite cases, not a proof that every resolution or mass method fails. Positive fallback and alternative stable decompositions remain viable. T110 exact-mode comparisons can use this evidence; its recipe-mode bias ordering remains uncertified because those recipe cones were not covered by this replay.
