# T124 bounded handoff

Claim df82455 preceded implementation; arithmetic-rescue scope64dc12f was
pushed before that follow-up. Current result commit e56531e. Only this
approach folder changed; T116 source and original pilot are pinned/read-only.

The first batched radial implementation changed7/9309 metadata tuples by
one rounding unit. Its predeclared scalar-radial/vector-trial rescue preserved
all metadata/values and30 threshold fixtures, but made the full64-cone pilot
18.2% slower. Both negatives are retained, with independent source review
9f3681a and original-owner review.

A separately frozen intermediate audit showed all sums/norms/projections/
sign bounds agreed exactly; only Python float `**2` versus the array square
differed. Scalarizing that one operation restored all9309 radial tuples.
The one-expression pinned adapter preserves every complete control and
threshold decision; owner56 and T55 independently reviewed its source and
attribution before the fixed64-cone pilot.

The rescue preserves all64 per-cone values and statistics exactly, including
3641 fallbacks,58256 nodes and partial mass .003088193770479861. Five fresh
alternating timings each yield medians .357795772 originalGrouped and
.351168585 rescue:1.85% lower local median. All12 calls cost4.425CPU.
This marginal local timing is not a portable speed claim or justification
for a full integration. The conservative arithmetic allowance increases
265021914 to272830302; there is no FLOP-saving claim.

Reproduce with the project interpreter, one thread,2GB and review lock:
`radial_probe.py`, `check.py`, `pilot.py` reproduce the initial variants;
`attribute.py`, `pow_check.py`, `pow_pilot.py` reproduce the isolated rescue.
Old artifacts are not overwritten by rescue scripts. Real-arithmetic bound
monotonicity and finite bitwise tests are not a universal floating/platform
equivalence certificate. Native or other positive primitives remain open.

DRI registers final classification after independent final cost/timing review.
