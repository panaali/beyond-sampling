# T110 class families, nearest facets and positive-volume ties

Independent algebra/source review of `class_region_families/families.py`
and the pinned `_cell_error` implementation through781c5eb.

Fix distinct top index a and second index b. The inequalities
`xa-xb>=0`, `xb-xk>=0` for all other k form a tree of T-1 independent
differences, whose common kernel is the constant vector. Adding either
`x0>=0` or `-x0>=0` gives T independent linear coordinates. The map to
these coordinates is invertible, and the family cone is the inverse
image of an orthant, hence pointed and simplicial. There are exactly
2T(T-1) such cones, with disjoint interiors; continuous Gaussian input
assigns zero mass to their shared boundaries. Explicit stable ownership
is still needed for deterministic boundary test points.

This is an exact classifier-integral decomposition. It need not preserve
an approximate recipe whose reference point depends on the nearest true
class facet. All difference-facet normals have norm sqrt(2). The nearest
top facet condition is `xa-xb<=xb-xk` for every k, equivalent to the
source's `2xb-xa-xk>=0`. Otherwise the nearest lower facet is indexed by
the largest remaining input c: require `xc>=xk` for remaining k and
`xa+xc-2xb>=0`. These regions partition each family, aside from ties.
Every part has an open interior and arbitrary common translation permits
both signs of x0. Thus the explicit nearest-facet construction has
2T(T-1)^2 pieces, cubic rather than quadratic. At T3 this is24, matching
the older half-chamber count.

This is not an impossibility theorem for quadratic descriptions. A family
can retain `min_k(xb-xk)` as a piecewise function without enumerating its
pieces, or group tied pairs with an existential condition on a higher
index. Such forms retain internal piecewise/nonconvex work. The exact
mass identity alone does not prove that a representative-based recipe
or its measured cost stays unchanged.

The pinned `_cell_error` mishandles an identically zero truth-minus-other
margin when the competitor index is smaller than truth. On the remaining
cone q, this entire cone is then error by first-index ownership: add
mass(q) and terminate. A higher-index exact tie can be skipped. Previously
removed error pieces must not be counted again. This local repair is
valid only for coefficient-exact zero; no tolerance or new near-zero
classification follows. The existing all-gates-off branch already uses
the correct constant-label rule.

The saved zero/identical-output failures therefore identify an inherited
implementation defect, not evidence against the class-family partition.
Finite recipe shifts on trained controls concern the changed approximation;
they do not refute exact integral invariance or all grouped alternatives.

## T4 constant-readout mass-rule discriminator

The later T4 control keeps geometry fixed while refining the inherited
Plackett mass rule. Its ordered-pair/sign chart probabilities are
independent exact controls: a specified top index is positive with
probability15/16, the second index with11/16, and either unspecified
remaining index with average(5/16+1/16)/2=3/16. Each ordered top pair has
probability1/12. Negative signs use complementary probabilities.
These justify the control's chart comparisons independently of the
network enumeration. The recorded default constant-output failure
and higher-order convergence concern finite mass quadrature, not a
failure of the class partition. They do not certify the default rule
on subsequent trained cases; those comparisons remain diagnostic.
