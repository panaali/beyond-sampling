# Pointed three-dimensional second moments

Contract: at most four homogeneous inward normal vectors in R3. Exact input
Fractions define the geometry. Positive-proportional duplicates and zero
constraints are removed. Normal rank below3 remains explicitly unsupported;
it can contain positive-volume lineality and is never silently called zero.

Full normal rank implies a pointed cone. Its extreme rays arise from feasible
signed cross products of independent normal pairs. Positive ray duplicates
merge. Under this pointed premise, ray rank below3 implies zero Gaussian mass.
For a full-dimensional cone, each actual facet has two extreme rays; constraints
with at most one active extreme ray are redundant for surface integration.
The sum of all extreme rays is interior. In a positive linear section, each
facet edge joined to the interior point gives a disjoint triangle fan. Exact
positive determinants, two-ray facets and degree-two vertex incidence are
checked before the spherical solid-angle formula is evaluated.

For an inward unit normal n, Gaussian integration by parts gives
M_C=P(C)I+(1/sqrt(2pi)) sum n m^T, where m is the tangent standard2D Gaussian
first moment over the facet. If its extreme rays are r,s, their inward tangent
unit boundary normals are normalized projections of s onto r-perp and r onto
s-perp. Hence m=(t_r+t_s)/(2sqrt(2pi)) and the boundary coefficient is1/(4pi).
The total is symmetric, and trace(M)=3P(C), even though each boundary outer
product need not be symmetric. For output vectors v,w positive on the cone,
v^T M w is their raw ReLU product there. No network assembly is in this phase.

Independent controls include orthants, exact orthogonal transforms, positive
normal scaling, duplicates/redundancies, wedges times a positive axis, and
C={x>=abs(z),y>=abs(z)}. For the last shape sign/rank symmetry givesP=1/12;
Mxy=integral phi(z)^3 dz=1/(2pi sqrt3); Mxz=Myz=0. Integration by parts gives
Mzz=P-(1-1/sqrt3)/(2pi), and trace/symmetry giveMxx=Myy=P+(1-1/sqrt3)/(4pi).
These formulas are independent of the spherical fan/boundary implementation.

Finite80/120-digit computation and160-digit analytic comparisons are not
outward certificates. Signed boundary cancellation, angle/determinant
conditioning and complete MP/native/geometry costs remain open. A finite
control failure is not a broad second-moment impossibility.
