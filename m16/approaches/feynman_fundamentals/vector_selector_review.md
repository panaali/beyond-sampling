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
