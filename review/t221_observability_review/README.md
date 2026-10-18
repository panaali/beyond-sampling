# T221 observational adapter source review

## Question
Does the local wrapper preserve the original scoped estimator's numeric behavior while exposing per-call candidate counts and partial failure information?

## Method
Read-only review delegated by renewed TASKS claim dbb5a068. Source preparation 33306831, published 327ec9e9, is in recover_56's review/srb_draw_diagnostics. Compare its selected AST nodes and injected constants with the already-reviewed T204 source/extraction. Review the proxy, draw wrapper, immutable snapshot and exception lifecycle separately. No source imports or calls.

## Status
Preliminary source/proof review passes within the stated ABI. The final finite control harness/protocol has not yet been reviewed; this note is not execution clearance.

## Results
The extractor preserves selected original function/constant nodes in their order. COST_Q=1.6 agrees with the frozen original setup. The wrapped estimator executes once; the RNG proxy forwards each shape request and returns the same array, and the draw wrapper returns the same source tuple. The original estimator return object is returned unchanged. Metadata operations can alter timing/resource behavior and are explicitly outside numeric cost claims.

The proxy records requested shapes before forwarding. Successful return shape and failed request error are separate; no count of a failed RNG's internal partial generation is inferred. Source drawn/kept/cost stay None when not observed. Successful draw output is recorded before subsequent Q work. The finally block creates nested immutable scalar/tuple records and clears live state. A nested call rejects before it can replace outer state. The source cache remains in the one extracted namespace across calls.

Require final controls to cover ordinary parity against preserved baselines; errors before/after a returned batch and after draw; cache/immutable-snapshot persistence; and nested rejection caught inside the outer callback so the outer call's state can still complete. Preserve raw records before assertions. Malformed/introspective RNGs, concurrent calls and failures of metadata allocation remain outside equivalence.

## Compute accounting
Only source/text inspection. Zero estimator, RNG, Q, candidate generation or model calls. No runtime or FLOP total inferred. The final planned controls require their own published source/resource gate under 20 CPU seconds and 2 GiB.

## Claims
- [inferred, source] Under successful metadata operations and the fixed ndarray RNG ABI, wrappers leave original numeric calls/objects unchanged.
- [inferred, source] Completed candidate counts can be observed without claiming unavailable partial RNG internals.
- [speculative] The frozen controls will verify the supported lifecycle and baseline cases; no run has occurred under this review.

## Alternatives not ruled out
A changed output could arise from numeric source changes or callback/metadata effects; exact AST/source pins and saved fixture parity discriminate them. Matching finite outputs alone does not prove arbitrary-RNG transparency; the explicit ABI and exception controls restrict that inference. A real cost-accounting defect remains separate from observability, and the wrapper does not repair policy.

## Failure modes
Observer work can fail or add overhead; introspective RNGs can notice a proxy. Snapshot immutability assumes the pinned scalar cost and tuple-of-scalar shapes. Runtime identity, general Gaussian rejection rates and complete estimator costs are not established by this source-only phase.
