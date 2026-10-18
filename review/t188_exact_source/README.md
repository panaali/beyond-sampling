# T188 exact-mode source gate

## Question
Does the fixed four-chart exact pilot change only the intended mode/flag and
retain the reviewed stop, provenance, diagnostic and resource contracts?

## Method
Explicit source-review assignment in T188 claim52634aed. Text-only comparison
of8e72f533 against prior capture/common and full_count/pilot; independently
apply listed substitutions and assert byte identity, check all protocol hashes.
Read exact recursion, TOL decisions, stopping, stack and output logic. No imports
or execution of author code. See source_gate.json for immutable pins.

## Status
Source gate passes for fixed H8 parents0,1,20,21 only. Owner must bind completed
review artifact before execution. Saved output still needs independent review.

## Results
Five common-module substitutions and three pilot substitutions exactly explain
the new source. Every recursive stop asserts exact=True/MODE exact; exact split
branch resolves cutting gates recursively. No-lookahead guard added, depth3
stop and forbidden final/mass callbacks retained. Existing caps/provenance and
arithmetic source unchanged. No numerical result inferred.

## Compute accounting
Read/hash/text comparison only; CPU in source_gate.json. No geometry, model,
chart, evaluator, mass or random/reference calls. Runtime budget is not priced.

## Claims
[inferred] The requested exact-mode source route and stopping contract are
preserved. [not established] Real-arithmetic partition completeness, feasible
merging, label-only equality, probability savings and full cost performance.

## Alternatives not ruled out
Floating TOL/clipping may differ from real geometry. Exact counts may differ
from recipe counts. Prefix piece/history repetition need not give useful unions.

## Failure modes
Changing MODE without initial exact flag; treating zero returned partial error
as an accuracy estimate; calling depth3 prefixes final cells; inferring a
certificate or cost win from a finite source-compatible census.
