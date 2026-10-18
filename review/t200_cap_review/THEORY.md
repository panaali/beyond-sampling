# T200 independent represented-cone cap theorem

Assignment804f5456 delegates this complementary theory/source/saved-control review to worker78. This note proves the proposed mathematical bounds only. Author source, execution and saved controls have not yet been reviewed; no numerical routine or experiment was run here.

Let the ambient dimension d be2,3,4 or5. Interpret every finite stored binary ray exactly as a rational vector r_i. Assume every ray is nonzero, a=sum_i r_i is nonzero, and a·r_i>0 for all i. Define

q=min_i (a·r_i)^2 / ((a·a)(r_i·r_i)), c=sqrt(q), delta=1−q.

Exact Cauchy–Schwarz gives0<q<=1. For every nonzero positive combination x=sum_i lambda_i r_i with lambda_i>=0,

a·x >= c||a|| sum_i lambda_i||r_i|| >= c||a||||x||.

The first inequality uses the positive unsquared inner products; squared ratios alone would lose their signs. Thus the represented positive hull lies in the spherical cap with axis a/||a|| and half-angle theta=arccos(c)<=pi/2. No unit-ray normalization is necessary. The origin is a null event for an isotropic continuous Gaussian. Radial symmetry makes its cone mass equal to the cap's normalized spherical area. Any missing true rays/lineality or erroneous floating H/V geometry is outside this containment assertion.

For d2 the cap mass is theta/pi=asin(sqrt(delta))/pi. Convexity of asin on[0,1] puts its graph below the endpoint chord, hence mass<=sqrt(delta)/2.

For d3 the axial coordinate is uniform on[-1,1], so cap mass=(1−c)/2<=delta/2, using1−c=delta/(1+c)<=delta.

For d4 the axial density is(2/pi)sqrt(1−z²). On z∈[c,1], sqrt(1−z²)<=sqrt(delta). With pi>3 and1−c<=delta, cap mass<=2delta sqrt(delta)/3.

For d5 the axial density is(3/4)(1−z²). On the same interval it is at most3delta/4, so cap mass<=3delta²/4.

All cases can be clamped above by1/2 because the cap lies in a hemisphere. If q=1, delta=0 and the represented cone lies on a ray; its Gaussian mass is0 for these ambient dimensions, so a zero bound is legitimate in this exact degenerate case. A rational upper enclosure u>=sqrt(delta) gives rational final upper bounds u/2,delta/2,2delta*u/3,3delta²/4. Rounding any input/intermediate downward without an enclosure is not justified.

Unsupported dimensions, nonfinite or malformed input, zero rays/axis, nonpositive inner products or resource/bit-limit failures may return the conservative probability upper1. This is an explicit fallback, not certification of a valid cap. Empty/lower-dimensional inputs need declared handling; fallback1 remains safe, though loose.

A binary64 rendering is a separate output. Validate its exact represented rational value against the rational bound; round outward or fall back to1, including subnormal and zero cases. Merely calling float() or betainc does not establish the direction of rounding. Never replace the rational certificate by its decimal display.

The proof certifies only the supplied represented positive hull under the declared isotropic law. It does not certify the neural cell's geometry, disjoint traversal coverage, retained-recipe approximation, other mass integrals, trained performance, arithmetic cost or runtime competitiveness. Finite controls can test implementation consistency; they do not replace these premises.
