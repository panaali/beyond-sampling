# T209 two conservative bounds on the same represented four-ray cone

Let R be an invertible4-by-4 matrix of exact stored binary64 ray rows. Its
positive hull C={R^T c:c>=0} has standard four-dimensional Gaussian probability

    P(C)=|det R|/(2*pi^2) integral_Delta (y^T R R^T y)^(-2) dy,

where Delta is y_i>=0,sum_i y_i=1, parameterized by its first three coordinates.
Indeed c=s y has Jacobian s^3 and integral_0^infty s^3 exp(-q s^2/2) ds=2/q^2;
the Gaussian normalization is(2*pi)^(-2). Delta's coordinate volume is1/6.
There is no Euclidean surface-volume sqrt4 factor in these coordinates.

A saved exact KKT certificate y*,a gives y*>=0,sum y*=1, a=y*^T G y*>0,
G y*>=a1, y*_i((Gy*)_i-a)=0, with G=RR^T. For any y in Delta,
q(y)-a=(y-y*)^T G(y-y*)+2(y-y*)^T Gy*>=0. The second term is nonnegative
because (Gy*)_i>=a, y>=0, and y*^T Gy*=a; the first is nonnegative since
G is a Gram matrix. Thus

    P(C) <= |det R|/(12*pi^2*a^2) < |det R|/(108*a^2),

using pi>3. Therefore B_det=min(1,abs(det R)/(108*a^2)) is a rational upper.
We reuse the already independently checked64 exact original-ray certificates;
no new support search, KKT solve or numerical integration is needed. Exact
certificate values must be bound to the same ordered ray bytes. B_det is a
geometric diagnostic, not a claim that certificate construction is free/cheap.

B_cap is the unchanged reviewed T200 primitive applied to the original rows,
without renormalization or axis changes. Both rational bounds are converted
outward using its checked exact-dyadic conversion. Every resource/domain
fallback remains1. Each bound is valid for the represented positive hull only.

Saved GL32 reference values and their integration/determinant-only bounds are
reused as descriptive numerical comparators. They do not include all kernel
floating error or geometry correctness. Ratios to these stored values are
therefore diagnostics, not rigorous tightness intervals. An observed disagreement
with a conditional reference interval must be retained, not silently fixed.

No completeness/disjointness of enumerated cones, classification replacement,
retained-recipe error, full pruning error budget, population or runtime inference
follows. The64 frozen evenly spaced indices are the existing T193 pilot, not a
new performance-selected subset. Results cannot be extrapolated to all37739
cones or other dimensions/models. This experiment tests finite practical
looseness of two bounds and does not retire either method family.
