# T106 degree-three edge algebra and circuit ceilings

Read-only review of `t6_edge_algebra/primitive.py` and `COST.md` through
b98bd20. The independent T104 sensitive-case check is preserved separately;
no full T6 integration is repeated here.

For unit edge endpoints u,v, c=u dot v and s=norm(u cross v), let
theta=atan2(s,c). In the midpoint frame the first coordinate along the
arc is A cos(t)+B sin(t), where
`A^2=(u0+v0)^2/[2(1+c)]` and
`B^2=(v0-u0)^2*(1+c)/(2s^2)`.
Symmetry gives `I2=A^2*(theta+s)/2+B^2*(theta-s)/2`. The p3 spherical
moment uses only m1 and this I2, so deleting the unused solid angle is
valid. Safe sine denominators followed by explicit zero-edge rejection
prevent a coincident edge from being accepted through division by zero.

The small-angle polynomial for theta-sin(theta) ends at theta^11/11!,
with absolute remainder at most theta^13/13! for the stated theta<0.1.
The propagated series-only bound is
`sum |nu0| B^2 remainder /(8*determinant*delta^3)`, as implemented.
It does not bound rounding in reconstructed unit rays, the raw/normalized
s,c pair, or the cancellation-prone final spherical moment. Both edge
variants therefore retain the original heuristic screens and positive
fallback qualifications.

## Independent line-group audit

The proposed ceilings use an arithmetic-circuit convention, not processor
instructions: basic arithmetic, abs, comparisons and Boolean operations
cost one; square roots and transcendental calls cost twenty; fixed small
integer powers expand into multiplications/reciprocals. Under this
convention a source expansion fits the stated ceilings:

* Endpoint reconstruction/validation fits 192 per triangle. The two 6D
  projections, perpendicular and three endpoint norms, resolution and
  reconstructed coordinates fit 208.
* Three 3D ray normalizations, cross products/norms, dot products, atan2
  and conormals fit 320. First moments fit 32; unused solid angle fits 64.
* Raw half-angle squares, all-three-edge small-angle polynomial through
  theta^11, theta^13 remainder, boundary and absolute sums fit 416.
  Three optional pair normalizations add at most 96. The original twelve
  trigonometric calls and remaining work fit its 640 allowance.
* Final scaling, range/cancellation checks and per-triangle numerical
  reductions fit 192. The 512 per-cone frame and 2048 per-batch setup
  ceilings cover fixed dimension checks, setup and header reductions.

All input triangles are charged these ceilings even if rejected before
the analytic branch; unchanged positive fallback/rule work is additional.
This finds no omitted numerical kernel in the primitive under the stated
convention. It is not a mechanical compiler proof or a full-estimator
cost certificate. Array allocation, interpreter/library dispatch and
actual execution remain measured separately.

The actual deleted analytic calls are one unused atan2, nine sine and
three cosine calls per attempted analytic triangle. Most reduction from
the historical 10000-per-triangle allowance is an accounting refinement,
not a sixfold algorithmic speedup. Even the differences between the new
ceilings include unequal conservative slack and are not exact physical
operation savings. Planning, geometry, rule construction and fallback
costs are still required before any end-to-end equal-cost comparison.
