# T268 fixed blind-sector saved contraction source review

## Question
Can the retained exact H4T4 simplex enclosures be safely grouped by sign orthant to test a sector-only binary correctness indicator?

## Method
Read sector.py, fixtures.py, run.py, PROTOCOL.md, protocol.json and launch.sh, plus inherited check_t4.py/check_enclosed.py/enclosed_mass.py source. Hash scientific artifacts without decoding them. The per-simplex enclosure source iterates the exact archived correctness-simplex sequence and records quadrature endpoints and absolute truncation upper bound in the same order; the application binds that compressed archive digest and5058-row coverage.

A nondegenerate4x4 ray matrix with no mixed coordinate signs and no zero coordinate column generates an open interior with that unique sign pattern. Coordinate zeros on individual boundary rays do not invalidate the orthant. The24-term exact determinant and product of ray L1 norms reproduce normalized absolute determinant independently of the saved determinant calculation. This certifies sector assignment for each represented simplex, not global correctness-region completeness/disjointness.

Independent centred nondegenerate Gaussian signs give mass1/16 to each strict orthant; zero-coordinate boundaries are null. Masks0,1,2,4,8 therefore have combined mass5/16. This is an input-event probability, not a network error rate or a T10 statement.

For each saved binary64 quadrature enclosure and truncation bound, exact Fraction subtraction/addition yields a conditional probability interval. Intersecting a simplex interval with[0,1/16] is justified by nonnegative mass and orthant containment. Sum intervals exactly within each sector, then intersect with the sector's known mass. Empty intersections fail rather than clamp to a point. Raw unphysical intervals and sums remain saved for diagnosis. The aggregate reference containment check tests consistency with its wider historical floating sum, not new reference validity.

For c in[L,U] within[0,p], error mass lies in[p-U,p-L], and conditional accuracy in[L/p,U/p]. The best constant binary-correctness decision has unconditional sector loss f(c)=min(c,p-c). The exact minimum over[L,U] is the smaller endpoint value; maximum is p/2 when the interval crosses p/2, otherwise the larger endpoint value. Summed sector loss intervals remain conservative without assuming independence of their enclosure errors. Strict L>0 and U<p proves both correct and incorrect positive-mass subsets under inherited geometry/numerics premises.

## Status
The source_gate.json binds the final pre-run repair and documentation clarification. Initial f875b6eb was amended to save row/global/sector calculations before assertions or subsequent calls and to require hexadecimal publication identifiers. Numerical formulas, fixtures and tolerances were not changed. One combined controls-plus-saved-contraction run requires published review and immutable execution gate.

## Results
[inferred] Four literal orthant cases, four geometric guards, six exact loss-law cases and three malformed-law guards cover the stated17 controls. They precede all scientific archive decoding. Target coverage is5058 rows in20chunks, all16 sectors and fixed five blind sectors;76 events,5066 orthant attempts,25 law attempts.5063 determinant evaluations correspond to121512 permutation terms, including the singular fixture but excluding guards rejected before determinant work.

## Compute accounting
One review-lock process,17 internal CPU seconds,19soft/20hardCPU,2GiB,25wall after lock and one numerical thread. No producer/reference mass evaluator, NumPy, model/checkpoint, RNG, face-law or geometry generation calls. Exact determinant and aggregation work is charged review computation; inherited reference construction/advice is disclosed, not free estimator work. Runtime resolved Python is pinned by this review in addition to the source runtime check.

Fresh output/log guards prevent overwrite; returned controls and each active target row are saved before assertions. Completed rows flush once and chunk hashes bind their ranges. Raw global sums and prospective sector intervals are retained before containment/physical-law checks. Failed attempts remain incomplete, with no retry or resource extension.

## Claims
[inferred] This is only a sector-only deterministic correctness-indicator test. It does NOT establish a predicted-class quotient obstruction: the true second-rank label varies within a sector, so correctness can vary even if the network prediction is constant. Conditional predicted-class laws and the richer T268 objective remain unresolved. No additional fixed-output baseline is silently added to this run.
[inferred] Completeness, disjointness, first-index correctness and outward reference/platform semantics remain inherited. Source/model byte hashes do not reproduce historical runtime. No cheap probability estimator, general compressed-representation impossibility, noisy-model transfer or ARC/HP1 result follows.

## Alternatives not ruled out
A compact conditional probability law can exist despite mixed binary correctness. Full predicted-class laws may be simpler than the correctness partition. Other sector summaries or learned-state information may change the result. The retained reference's construction costs can dominate any hypothetical cheap aggregation.

## Failure modes
Incorrect archive alignment or old geometry/reference assumptions would invalidate semantic interpretation despite arithmetic consistency. Physical clipping cannot repair empty intervals. Constant-correctness conclusions must use strict interval interiors, not midpoint estimates. Independent saved auditing remains required after this fixed run.
