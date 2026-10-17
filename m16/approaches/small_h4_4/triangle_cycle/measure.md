# Independent composition and error check

The standard Gaussian density in four dimensions is exp(-||x||^2/2)/(4*pi^2).
Writing x=r*R*y, with sum(y)=1 and r>=0, gives Jacobian |det R|*r^3.
The radial integral is 2/||R*y||^4, so the cone probability is

    |det R|/(2*pi^2) integral_Delta3 ||R*y||^-4 dy.

Set y0=u and yi=(1-u)*zi for i=1,2,3, where sum(z)=1. The barycentric
Jacobian is (1-u)^2. The inner triangle endpoints are

    Ai=u*v0+(1-u)*vi, i=1,2,3.

For E=[A1,A2,A3]=Q*B with Q orthonormal, the radial projection z->omega
of B*z has solid-angle Jacobian |det B|/||B*z||^3. If c satisfies
B^T*c=1, then ||B*z||=1/(c dot omega). Thus the triangle norm^-4 integral
is integral(c dot omega)dOmega/|det B|, with no additional area factor.
This independently checks the formula in T55 triangle_theory.md. The
coordinate simplex has area1/2; the orthogonal endpoint control is3*pi/4.

Every endpoint lies in the common signed-L1 plane s dot x=1. Therefore
c=Q^T*s, avoiding the inverse solve, and ||c||<=||s||=2. A tangent-basis
implementation uses the plane distance delta=1/||c|| instead. The factor
is moment/(det(B)*delta), agreeing with the exponent4 batch API.

For each original cone choose a<=q=||R*y||^2<=b using the unchanged Gram
bounds from stable_volume.py. Approximate q^-2 uniformly by a degreeK
polynomial in q with tail E. Under the above parameterization it has
degree<=2K in u, and multiplying by (1-u)^2 raises this to2K+2. Therefore
n-point Gauss integrates the reduced expression exactly for K<=n-2.
The exact inner triangle operator is positive. Both the true measure and
the composed positive quadrature have total mass1/6. Their polynomial
remainders contribute at most E/3, giving the unchanged cone probability
bound |det R|*E/(6*pi^2).

When a triangle uses numerical fallback, its absolute error bound is
multiplied by |det R|/(2*pi^2)*w_j*(1-u_j)^2. The batch primitive exposes
the sum of inner bounds, so the driver conservatively multiplies that sum
by the largest associated positive weight in the batch. This is looser
than individual weighting but remains valid in exact arithmetic.

All statements concern integration and exact geometry/arithmetic. Floating
clipping, determinant errors, and spherical-moment cancellation are not
certified. A heuristic guard does not convert this into a complete accuracy
certificate. No random model-input evaluations enter this calculation.
