# T242 source boundary for the owner T239 question

Claim 9628bc97. This review reads source and Git tree metadata only. It does
not decode any NPZ/checkpoint, import the author, or recompute any model/mass.
`manifest.json` pins the freshly fetched author heads, selected source snapshots,
tree inventories, directive and completed prior total-coverage handoff.

## What the available T206 leaf archive represents

`anatomy.py` wraps the original `_cell_error(c,g,A,Wo,truth,ops)` callback.
It records the total error mass `e` from output-margin clipping of one terminal
activation leaf, then separately evaluates that entire leaf's mass `lm` using a
throwaway counter. Its NPZ fields are `err,mass_ops,simplices,kind,leaf_mass,rays`.
Here `rays` is the NUMBER of rays, not saved ray coordinates. It does not save
the per-output clipped pieces, original inequalities/ray matrices, input-gap
values, chart/leaf IDs, gate histories or a proof of uniform correctness.

The callback precedes the output error partition, so an activation leaf can
contain both correct and wrong inputs. One index can legitimately contribute
`e` to error mass and `lm-e` to correct mass. Sorting those two contribution
vectors measures coverage by ORIGINAL activation leaves, not counts of disjoint
uniformly labelled decision regions. It is useful if named accordingly.

`kind=0` is selected by zero charged mass operations; `kind=2` is selected by
`abs(e-lm) <= 1e-12*max(lm,1e-300)+1e-18`; all others are `kind=1`.
These are source diagnostics. Especially the near-equality test is not an exact
uniform-wrong certificate. A small positive correct remainder could be hidden by
its tolerance. Neither kind nor a leaf-center prediction should replace the
two mass contributions. The pinned source's handling of zero affine readout
margins must retain existing tie-review qualifications; no new trained defect
is inferred here from the absence of an explicit tie branch in this copy.

Leaf mass is diagnostic work, omitted from the estimator's main operation
counter. Coverage curves that require it, sorting or future gap intersections
must report that preparation separately; they are not a free causal explanation
of estimator runtime. Source `mass_ops` is an inherited proxy, not an independently
complete computational cost. `anatomy.py` has both factorial chamber and quadratic
truth-cone starts, which produce different partitions. Do not mix their counts.

`anat_summary.py` sorts error masses using the already computed per-leaf errors
to propose an oracle skip budget. This does not implement a cheap certified rule
for finding those leaves before integration. T239 can reuse the raw masses for
a descriptive coverage question without inheriting a cost-saving claim.

## T192 availability is corpus-specific

The frozen GPU tree has run summaries and a separate `export_simplices.py`.
The pinned estP.py routes _cell_error output simplices into the returned mass
collector; optional full-leaf diagnostics use a separate collector. That export
calls the default without leafsum, keeps every STRIDE-th H8T5 error simplex and
saves its rays and two numerical mass evaluations, with total/stride metadata.
It is not a complete collection of correct and wrong activation leaves, nor a
random sample with known probabilistic coverage weights. It cannot substitute
for the requested H16T5 full correct/error corpus. Other author run summaries
are not decoded here and cannot establish that missing corpus on their own.

The current tree inventory contains no dedicated T239 implementation identified
by this review. This is a frozen published-tree statement, not an assertion
about unpublished code. The author may have new archives elsewhere; a later
audit should pin those exact bytes and producer before using them. No capture
or model replay is requested by this source review.

## Reuse boundary

T231 independently reviewed total prefix-mass coverage, including its numerical
and provenance limitations. That does not certify T239 correct/error membership,
terminal uniformity, gap-conditioned masses or a larger-T completeness claim.
Use the prior handoff as a prerequisite, not as a substitute for those checks.
