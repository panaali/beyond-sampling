# T264 storage/parity source review

## Question
Does the frozen storage rewrite preserve the exact frontier output, and does its bounded parity harness test that without rerunning the old helper?

## Method
Read source7d5856cd, claim78a542ed. Verify every protocol hash, identical fixtures, and literal primitive transformation using `check_metadata.py`; parse source without importing it. Inspect added prior-result/reviewer binding, full raw-field equality, strict zipped coverage and exact rejection-record equality. No new helper or retained target payload is evaluated.

## Status
Source/resource gate passed for one fixed12-valid+6-guard parity run, conditional on published execution scope and immutable gate. Independent saved parity remains separate.

## Results
[inferred] At iteration k, the old chosen list is exactly ids[:k]. The unchanged predicate updates best only for feasible strictly greater savings, so best_len retains exactly the length at the same last update; no update leaves it0. ids is unchanged throughout the scan. The final ids[:best_len] equals the former copied indices, including empty selection, zero costs and first tied maxima. Arithmetic, branch decisions, emitted prefix/breakpoint/80-decision records and order are unchanged.

[inferred] Repeated best-index copying is removed; a single final slice is used per method. This is a source storage argument, not measured runtime or full complexity/cost evidence. Exact Fraction computation and all record retention remain.

[inferred] The harness compares all raw fields against the already independently audited old12 records and compares all6 rejection records. It calls only the new helper18 times; no old helper replay. Old raw and saved-audit digest bindings and complete/pass_all requirements precede calls. Returns are saved before assertions; failures clear completion flags. Twelve cases do not replace the general loop-invariant argument.

## Compute accounting
Metadata hashing, textual transformation checks and AST parsing only. No primitive, model, target array, geometry, mass, CDF or RNG calls. One permitted future run:20/20CPU seconds,2GiB/2GiB address space, internal17CPU,25sec timeout after reviewlock, thread variables1. The caller must refuse existing run log and preserve command/errors; no automatic retry. OS limits are checked in the harness before helper import.

## Claims
[inferred] The literal storage rewrite is semantically equivalent under the original valid-input contract. The frozen harness is adequate for bounded parity controls; it does not authorize retained-panel decoding.

## Alternatives not ruled out
Other implementation defects, target schema errors, inherited cost omissions and numerical mass errors remain possible. No estimator speedup, integer optimality or true-mass certificate follows.

## Failure modes
Mutating ids during iteration or changing strict-greater tie handling would invalidate the argument. A mismatched prior artifact or any output/guard difference must stop and remain preserved. Future retained-panel code needs its separate source/resource gate.
