# T82 network-topology versus coordinate discriminator

Read-only slices.py review confirms the fixed Gaussian input z and tie
displacement t give affine preactivations. Window inequalities, side-specific
target (R23 negative->b; R12 negative->a), zero-gate assignment and first-index
identical-readout ownership are consistent. CCW clipping preserves the
vertical-bound formula. Domain plus correct-polygon vertices suffice for the
integrated correct mass; irrelevant gate vertices are not required.

Gaussian-coordinate integration needs density phi(z)/(1-Phi(tie)) for the
upper coordinate R23, or phi(z)/Phi(tie) for a lower coordinate R12. Bounds
are the images of q=.05,.95. The quantity is an unnormalized .9-wide q
integral, so dividing by .9 would change it.

The independent driver reuses read-only polygon masses but assembles its own
Gaussian weights, with4 nodes per panel on exactly the eight saved cases.
Network knots and equal-width-z controls have identical panel/node counts.
Geometry-only panels have fewer nodes and are explicitly unequal work.
Errors versus the saved refined scalar reference:

|H|kind|configuration|network z|equal-width z|geometry-only z|
|---|---|---:|---:|---:|---:|
|2|23|0|1.05e-7|-7.13e-9|1.05e-7|
|2|23|11|2.36e-7|-4.34e-10|2.10e-7|
|2|12|0|5.61e-10|2.71e-7|-.002885|
|2|12|5|8.34e-8|-4.30e-5|9.12e-8|
|4|23|0|4.51e-8|-6.81e-7|-6.51e-7|
|4|23|11|-6.37e-10|2.89e-5|.001282|
|4|12|0|9.21e-8|-3.50e-5|-8.13e-5|
|4|12|5|9.21e-8|-3.89e-5|9.12e-8|

Network topology improves6/8 matched comparisons. Both H2R23 slices instead
favor equal-width Gaussian panels, showing that a coordinate change alone
explains their earlier large gains. Some other slices need the actual network
knots to avoid large errors; this is a finite discrimination, not a universal
rule for which representation to choose. Geometry construction and query
work differ despite matched numerical node counts, so no end-to-end cost win
is established. The4-node normalizer defect reaches9.37e-6; it is retained,
not repaired by renormalization. Refined positive quadrature remains the
reference, not a rigorous enclosure.

Cost .045400CPU after imports, same eight slices, no random inputs. This is
an independent rule assembly and source review, not an independently coded
polygon solver. Thin polygons, floating vertex ordering and nearly inverted
vertical sections remain qualifications. Raw source and outputs are
check_slice_coordinates_review.py and slice_coordinates_review_results.json.

Full-contraction orientation failure followup: the owner saved a triangle
whose first edge is (+1.11e-16,0) and third vertex lies.06363 below it. The
exact signed area of these represented floats is negative, about-3.53e-18.
Absolute-area retention admitted it, while vertical_mass assumes CCW order.
Reversing vertex order preserves the represented set and repairs this
precondition without deleting a sliver. It does not prove restoration of the
unrounded clipping result. Count reversals and areas, retain the failure,
and for polygons with more than three vertices check convexity as well as
total signed orientation. A rigorous accumulated roundoff mass bound needs
more than diagnostic floating sliver areas. Owner received this distinction
before the separately frozen fix; no production edits here.

The owner subsequently added exact Fraction shoelace evaluation for stored
float vertices near area cancellation. It recovers the saved clockwise
sliver even when the reversed floating area rounds to zero. The predicate
still converts the exact area back to float before testing sign; an extreme
unrepresentably tiny nonzero area could underflow. Therefore exact-sign/no-
sliver-deletion language needs that domain qualification or a separate
rational sign. Negative local turns are counted rather than rejected, so
full-run convexity counts must be inspected before extending the claim.
# Final halfspace and normalization followup

Read-only review of the final T82 `slices.py`, `check_partition.py`, and saved
results confirms that vertical sections now use inherited original affine
constraints. A constraint az+bt+c>=0 supplies a lower t bound when b>0, an
upper bound when b<0, and a feasibility test when b=0. Intersecting these
bounds and assigning zero to an infeasible interval is the correct operation;
it does not require the rounded polygon's edges to remain convex or oriented.
The registry retains a strong polygon reference alongside each identity key,
so identity reuse does not silently substitute another polygon's constraints.

This repairs the represented-section defect exposed by clockwise and
nonconvex slivers. It does not certify branch completeness: rounded vertices
still determine whether branches survive clipping and supply projected
quadrature knots. The Fraction fallback recovers the area sign of stored
floats in the saved fixture, not the exact original network polytope. Its
conversion back to float also leaves an underflow qualification for arbitrary
extreme examples. Failures and rational fallback work remain recorded.

The final partition discriminator computes M from the geometry-only domain
halfspaces, independently of the network's correct gate cells. Values are
M3=.998019014152934 and M5=.998687578953307, with true complete face mass1.
Replacing 1-E by 1-E/M improves three of the four fixed H2/H4 contractions
and worsens coarse H2. Normalized errors are approximately -.004920,
.000173, +.007404, +.003687. The raw results remain essential.

For complete configuration symmetry, a constant label has E=(1-1/T)M,
so normalized accuracy1/T holds algebraically even when the outer rule is
biased. This is a useful partition/tie bookkeeping control, not independent
validation that normalization removes network error. Correlation between E
and M errors determines whether the ratio helps. The prior matched-coordinate
check remains the distinct numerical discriminator: six of eight slices
benefit from network knots over equal-count z panels, while both H2 R23
slices favor the equal-count control. No universal topology or efficiency
claim follows. The adapter's cost proxy remains explicitly partial.

No new engine run was needed for this source/interpretation followup.
