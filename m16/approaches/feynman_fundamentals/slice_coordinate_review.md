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
