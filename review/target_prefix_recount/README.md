# T219 independent target-prefix recount

## Question
Do independent exact dyadic gate-history counts verify or correct the reported target M16T10 prefix counts34,792,21032 through depths2,3,4?

## Method
Own integer extrusion and double-description clipping in core.py, proof in PROOF.md. No estT/NumPy geometry, artificial simplex fan, floating tolerances, forward model, mass or sampling. Start with bounded analytic controls and an independently counted planar k2 slope partition before any deeper target enumeration.

## Status
Analytic controls independently saved-reviewed5e1664f7; target-byte/planar34 regions independently saved-reviewed13e3a70e. The separately source-gated depth3 phase completes with792 regions, saved review pending; see DEPTH3_RESULTS.md. No depth4/5 enumeration has executed; these require verified cheaper phases and separately frozen resources.

## Results
All18 frozen analytic controls pass once: seven geometric controls, eight small networks, three invalid-domain rejections. Exact analytic rays/counts, positive-volume certificates and scaling/reflection invariances passed. Raw records are controls_result.json and36 durable progress events in controls_progress.jsonl; controls_run.log preserves the launcher output (empty on success). Author CPU0.087718203 seconds; peak RSS25,820KiB under the strict20CPU/2GiB launcher in CONTROL_PROTOCOL.md. Independent5e1664f7 checked104 cones/94 states using a different exact active-normal/minor construction and exhaustive synthetic gate words; internal ledger was aggregated, not replayed.

The target checkpoint was decoded once with weights_only=True; all432 stored coefficients and the saved float64 archive bytes agree. Recurrence coefficients use exact common denominator549755813888. Two independent constructions give34 depth2 regions:16 distinct rational slope roots in each x0 halfplane, hence17 sectors each. All34 histories/ray pairs/hidden maps agree with the independent cone construction;36 depth1/2 exact witness certificates are saved. This agrees with the author's reported34 at k2. Raw target_k2_result.json, six durable progress events, and target_k2_run.log retain the full weights, slopes and certificates. CPU0.893409506 seconds before final serialization (0.899734043 after it), RSS197,828KiB, max watched integer183 bits. Reproduction command is in TARGET_K2_PROTOCOL.md; no rerun is needed for saved verification.

Author source prefix.py was read only to understand the comparison contract; no geometry implementation or numerical state reused. T201 pilot weights were independently rebound to the existing checkpoint bytes. TARGET_PLAN.md retains the prospective scope; completed raw target work is limited to depth3.

## Compute accounting
The18 analytic controls include construction, exact certificate validation and serialization time. Per-control named integer counts and peak retained bit sizes are saved; these exclude some intermediate bit sizes and are not a native/fullcost model. No model/MC/mass/RNG calls. Any later target enumeration remains separately bounded and reviewed.

## Claims
[inferred, source reviewed] Under stated complete-cone invariants, exact extrusion and edge clipping enumerate all full-dimensional gate histories. [measured, independently saved reviewed] The18 frozen controls pass. [measured, independently saved reviewed] Target depth2 has34 exact gate-history cones with two independent constructions. [measured, saved review pending] Depth3 has792 exact gate histories. [speculative] Deeper target counts will agree with the floating author counts; discrepancies will be preserved and diagnosed, not suppressed.

## Alternatives not ruled out
Floating tolerance effects, actual target-byte mismatch, redundant representations, exact arithmetic cost limits, saturation after k5, map equality across distinct histories and methods aggregating regions remain open. A finite count or resource cap does not establish whole-family impossibility.

## Failure modes
Confusing rays with columns, incorrectly treating lineality as a pointed cone, incomplete edge adjacency, sign-flipping primitive normalization, counting boundary-only histories, zero-form tie mishandling, and extrapolating finite counts to T10 are explicitly guarded or qualified.
