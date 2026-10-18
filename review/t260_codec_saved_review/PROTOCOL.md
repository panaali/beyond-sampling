# T260 independent fixed saved codec audit

Scope published in TASKS-only 1815463f before implementation. Producer source
44c7450e and schema were read; completion notification d30fe7ab/cb169ca1 arrived
during implementation with coverage/timing only. No producer numerical result,
packed artifact, payload JSON or NPZ array was decoded before checker freeze.

One stdlib saved audit of exactly eight artifacts and 32 controls. Independently
parse bytes with a bounded tag/gamma reader, reconstruct runtime AST schema and
source-only normalized programs, compare exact types and binary64 bits, bind all
68 arrays in the 23 retained inference archives using NPY headers and little-endian
struct decoding, and check dependency closures/full-column overhead. This includes
unselected fitted arrays as provenance overhead. No producer codec, representation,
feature builder, predictor, NumPy, fit, network, RNG or score evaluator is imported
or called. AST construction does not execute its code. The synthetic table-route
outcome is checked against its fixed exact arithmetic result; it is not replayed.

The packed format has leading zero padding, not trailing padding. The independent
reader checks complete consumption, each subtree bit length and grammar category
census. Saved payload JSON loses tuple/list distinctions, so that comparison allows
those two container types; a separate comparison against independently assembled
archive payloads requires exact tuple/list structure and exact float bits, including
signed zero. Source AST comparison omits location attributes as the declared code
requires. The saved codec guard is source-qualified; our fixed four-bit prefix
also verifies independent trailing-bit rejection without author code.

Limits: at most 80,000,000 bits per artifact and 240,000,000 total. One run only,
95 internal CPU seconds, OS soft105/hard110 CPU seconds, hard/soft2GiB address
space, 120 wall seconds after the primary lock, one BLAS thread. No automatic
retry, widening, truncation, quantization or alternative selection. The larger
fixed budget was accepted by h4 before implementation. Top-level stdlib imports
are outside the main timer but inside external limits. All validation, hashing,
NPY and packed decoding and saves inside main are reviewer work; this is not a
measurement of predictor deployment cost. Full process log retained.

The checker requires a published h4 source/resource review directly binding
check.py, reader.py, protocol.json, PROTOCOL.md and launch.sh, and a separate
completed_inputs.json binding final producer raw/progress/log/gate and all24
artifact files. Completion bindings are metadata-only hashes populated after
checker freeze, never after seeing scientific values. Separate maps are each
verified, so overlapping keys cannot override a frozen hash. Actual invoked and
resolved interpreter and binary digest are checked before decoding. Producer
source/protocol/manifest/gate/reviewer pin maps and successful flags are rechecked.
Prior reviewed pilot is reused, not refit or rescored.

Durability: refuse nonempty outputs and an existing log. Write atomic results
before each cap check; append per-artifact progress and preserve any exception as
complete=false/pass_all=false. Twenty reviewer events: prerequisites, controls,
archives, eight starts and eight checks, final completion. Producer's twenty
progress records must match fixed coverage and monotone CPU timestamps bounded
by final CPU. Preserve all resource or metadata failures separately; do not rerun
completed contractions.

This establishes a code-relative retained-artifact upper length, exact framing and
payload consistency. It does not certify predictor equivalence, minimal length,
fresh agreement, a population rate curve, full inference cost, or recovery of the
historical C2 models. Inherited pilot metrics remain descriptive only. T55 owns
producer source review; h4 owns this independent checker's source/resource gate.
