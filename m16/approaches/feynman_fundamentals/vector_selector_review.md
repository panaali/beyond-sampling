# T124 scalar-radial/vector-order review

Source4229e80 retains the scalar radial map after preserving seven exact
metadata mismatches from batched radial computation. For a_n=binom(n+2,2)
r^n, a_(n+1)/a_n=r(n+3)/(n+1)=q_n decreases in n. Once q_n<1, a_n decreases
and1-q_n increases; E_n=a_n/[mid^3(1-q_n)] therefore decreases. A passing
chosen permitted order and failing immediate predecessor establish real
minimality; a nonconvergent predecessor implies all earlier orders also
nonconvergent. r=0 has zero bound at the first permitted order.

The source guards finite positive midpoint, positive qmin and0<=r<1,
scalar-rechecks selected and predecessor using the original expression,
and falls back for vector nonpass, near-threshold/nonconvergence cases.
Queue reset uses finally. Accounting retains original allowances and adds
vector/check work. This establishes source consistency and real-arithmetic
logic, not a universal floating monotonicity proof; finite9309-case and
adjacent-tolerance bitwise controls support only their declared coverage.
The initial slower controls remain preserved. A fixed64-cone pilot may
measure CPU effects, but neither FLOP savings nor family failure follows
from these controls. No model evaluation or expensive pilot was repeated.

## Isolated scalar-power rescue and fixed pilot

The frozen attribution compares every intermediate on the same9309
triangles. Only seven squared projections differ: Python float power and
NumPy square round differently. `check_radial_pow_review.py` independently
reproduces all seven projections/power discrepancies and verifies the
hybrid radial tuple equals the original scalar tuple for each. The adapter
asserts the old selector hash and replaces exactly one radial setup
expression, retaining every guarded order decision and adding160 units
per triangle. Batch reduction equivalence remains finite-panel/platform
specific. Reviewer initial length-one-array float conversion failure is
preserved and repaired by explicit element indexing.

The unchanged64-cone pilot has identical complete rows/stats. Independent
raw timing medians are.357795772 baseline and.351168585 rescue, ratio
.981477738 (1.85% decrease); this small fixed local timing effect is not a
portable speed guarantee. Named conservative allowances increase from
265,021,914 to272,830,302, independently summed from their parts. All3641
fallbacks and58256 positive nodes remain; partial mass is unchanged at
.003088193770479861. Initial scalar-radial guarded18.2% slowdown and seven
radial failures remain preserved. No full integral, accuracy gain or FLOP
improvement is established, and this result does not exhaust other batch
implementations. Review reproduction plus raw audit costs.0185CPU.
