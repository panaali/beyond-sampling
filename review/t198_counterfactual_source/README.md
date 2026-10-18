# T198 complementary counterfactual source review

## Question
Does the pinned attribution code measure the estimate change from removing a single split in the adaptive recipe, or a gate intervention on fixed refined pieces? Does its guard log measure additional repaired-traversal splits?

## Method and status
Published assignment557b7f5a precedes this owned note. Read-only source reasoning on author6a2a5403 and h4's frozen snapshot1707472b, with memo3712ca42. SHA256 pins in pins.json. Review complete; no original or modified routine was imported or executed. This complements h4/T55 cap, saved-array and table audits; it does not repeat them.

## Findings
[inferred from source] The narrower fixed-piece interpretation in SOURCE_REVIEW.md is supported. attr.py::_finish (132–153) first obtains all returned(c2,g2,tags) from _margin_split. For each minority tag it retains the same c2, At, all other entries of g2, changes only gate k to the recorded frozen value, then invokes downstream _finish with a fresh Ops and restored STATS. The split geometry has already been applied; the code never rejoins the pieces or replays the parent with that original split disabled.

Let the completed subdivision be C=union P_l. The recorded signed contribution has the form sum_l(E(P_l,g_l)−E(P_l,g_l with k frozen)) on tagged minority pieces. The absolute statistic is sum_l|E(P_l,g_l)−E(P_l,g_l with k frozen)|. It is neither the absolute net effect nor a joint-pruning error. Multiple tagged interventions may refer to overlapping domains.

To identify that sum with a fresh adaptive estimator that removes the original split requires an additional representation/subdivision-invariance argument, and compatible gate assignments on the recombined parent. That premise is absent from this source. In estT.py::_ref (246–249), the reference is a mean of current boundary rays when sufficiently many qualify, otherwise all current rays. attr.py::_margin_split (68–128) uses this reference to choose frozen gates and uses current ray extrema, minority vertices and adjacent edge intersections for its trigger tests. Changing the subdivision changes those inputs to the approximation. Additivity of an exact integral over disjoint domains alone does not establish additivity of this geometry-dependent approximate estimator.

The source review does not prove an actual numerical mismatch between the two interventions. In a special case with exact downstream evaluation of one fixed pointwise function, suitable gate consistency and exact additive masses, refinement can be harmless. The saved operation remains a legitimate fixed-piece diagnostic. The author's wording 'first-order change ... if that single split were pruned' needs this qualification; it is not established as a literal parent-split-removal replay or derivative.

[inferred from source] attr.py::_guard_log (58–66) is called only at the no-trigger return (108–112). It records maxima of individual and summed shift ratios at a finite union of test points. The logged event is an original traversal location at which a summed pointwise inequality may fail. It is not a run of a repaired splitter: it neither selects a new split nor records the descendants/retests a repaired traversal would create. Consequently a ratio of these event counts to original splits is not an observed percentage increase in splits. A chosen repair could subdivide, improve its bound, evaluate a cell exactly or defer it; even a prescribed splitter could alter subsequent event coverage. Pointwise upper-bound failure is not proof of actual misclassification, and finite test-point success is not a whole-cell certificate across unresolved downstream kinks.

## Results and compute accounting
No numerical outcome, model evaluation, mass/geometry routine, new counterexample, sampling draw or candidate operation was run. Only text inspection and SHA256 metadata hashing. Original per-piece diagnostics, sampled flags and source caveats remain unchanged; no experimental result is invalidated merely by this qualification.

## Alternatives not ruled out
A direct fixed single-split-disabled replay, a proof of representation invariance for a restricted exact suffix, or a fully specified repaired traversal could establish stronger interpretations in a separately authorized phase. The current note does not authorize those experiments or claim a method-wide defect.

## Failure modes
Confusing changed cell representation with a proven numerical discrepancy; adding overlapping effects; calling event counts repaired-tree counts; equating a conservative bound's failure with actual error. All conclusions here are source-semantic and confined to the pinned implementation.
