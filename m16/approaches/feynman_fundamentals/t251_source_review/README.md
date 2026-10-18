# T251 independent written source review

## Question
Are the T251 source and oblique-projection qualifications supported by the actual T241 phase-A implementation?

## Method
Read five exact Git text blobs at a24a544c and the review package53da2f0c. Manually derive the projected-norm/removed-SSE distinction and counterexample; check suffix/donor semantics and source workload. check_metadata.py verifies byte/hash/AST bindings without executing author code.

## Status
Complete/pass for the written/source scope. No empirical score validation or execution clearance.

## Results
[inferred] For general oblique P, ||HP^T||²/||H||² differs from the removed-SSE fraction. The exact example gives1 versus-1, including from strictly positive states. Donors at h9 are conditioned on original final gates, not prefix-only information. Missing paired outcomes, selection and model/runtime provenance constrain interpretation. REVIEW.md gives qualifications.

Reproduce metadata only with `python m16/approaches/feynman_fundamentals/t251_source_review/check_metadata.py`; existing output blocks another invocation. Original metadata pass before the one-sentence clarification is retained separately. No failed review or numerical retry.

## Compute accounting
Text/hash/AST and written algebra only. Zero author imports, checkpoint/binary/result/log payload reads, model, sampling, fitting, GPU or numerical experiment calls. Source2830N+500000 recurrence-step count is partial work, not FLOPs or measured runtime.

## Claims
[inferred] Written algebra and source semantics are correct. No actual target energy, accuracy, causal role, redundancy, off-manifold explanation or runtime failure is confirmed.

## Alternatives not ruled out
Saved accuracies may be correct; projected-norm ratios may be informative under an accurate name. Independent donors, held-out pattern selection and paired tables could support stronger claims under new scope.

## Failure modes
Source-to-run provenance and finite eigensystem conditioning are unverified. Idempotence alone does not make a projector orthogonal. Same-pattern donors can still create signed/unreachable states. No missing joint outcomes can be reconstructed from aggregate accuracy/agreement alone.
