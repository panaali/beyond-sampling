# Pre-execution physical-gap mismatch

## Finding
The baseline source 7d874a2f does not implement its stated physical gaps {0, 1/4, 1/2, 1}. Its positive-gap predicates instead implement physical gaps {1/8, 1/4, 1/2}. This is a source-unit defect, not an evaluated failure of the method or its probability enclosures. No baseline archive or numerical result was decoded to find it.

The inherited T189 `THEORY.md` defines `v=2x`; the original boundaries `(-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8)` are in v units. The inherited `probability.py::setup` evaluates `cdf(F(v,2))`, and the enclosure recurrence uses the corresponding half-scale input. These facts bind both the box probabilities and network certificates to physical endpoints B/2.

The new baseline passes integer B endpoints into `pairs(box,g)`, where g is the integer quarter-gap numerator. Its containment comparisons use `4*(B_i-B_j)>=g`; its intersection tests use the same scale in every endpoint. Since B_i-B_j is twice the physical difference, the tested physical threshold is g/8. The advertised g/4 threshold instead requires `2*(B_i-B_j)>=g` and the corresponding change to all intersection endpoints. Zero gap is unchanged.

This argument identifies the represented event without decoding any of the 81 output or inherited archives. It does not yet independently validate the completed sums or prove that their original reported numerical values equal a corrected run. The author acknowledged the source mismatch and will preserve the raw run and original labels.

## Review preparation history
`check_initial_unrun.py` preserves the first, unexecuted checker draft. Its gap scaling initially mirrored the author's integer endpoint representation. A subsequent cross-check against the inherited physical-coordinate definition caught the mismatch before checker freeze, source gate or execution. The draft was never imported or run and has no completed-input or execution gate. Its comment about coordinate scaling is incorrect and remains preserved as evidence of that preparation error.

## Next scope
Do not overwrite or silently relabel the author raw data. A separately declared saved-only contraction could verify it as a diagnostic at physical gaps {0, 1/8, 1/4, 1/2}, retaining the failed original physical-gap contract. The missing physical gap 1 requires a distinct corrected and reviewed author phase. No rerun or new numerical phase is authorized by this finding.
