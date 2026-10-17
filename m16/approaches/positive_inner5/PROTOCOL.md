# T11589 fixed primitive gate

Claim 23bfb42e, Claude DRI. No estimator or source quadrature is replaced.

For q=az²+2bz+c, a>0 and delta=ac-b²>0, let ti=b+ai,
ri=sqrt(ti²+delta), si=ti/ri. The exact primitive is
a^(3/2)/delta²*(s-s³/3). Its difference factors as
ds*[0.5(delta/r0²+delta/r1²)+ds²/6], a positive expression.
For strict same-sign endpoints rationalize ds=delta V, where
V=a*(abs(t0)+abs(t1))/(r0*r1*(abs(t1)*r0+abs(t0)*r1)).
This cancels both explicit inverse delta factors analytically. Otherwise
ds=abs(s0)+abs(s1), including a zero endpoint. The constant D=0 branch is exact
in real arithmetic. Nonconstant collinear inputs are explicitly unsupported.

The implementation scales P,D by one common max absolute coordinate, computes
delta from ten squared wedge minors, and rescales by scale^-5. These steps do
not certify floating arithmetic. Scaling itself can underflow coordinates or
the final factor. Every unsupported/nonfinite/underflow outcome is retained.
No rule fallback, tuned thresholds, model/geometry calls or adaptive precision.

Alternatives: (1) endpoint subtraction causes the observed loss, discriminated
by same-sign tiny-angle fixtures against a direct floating primitive;
(2) Gram/determinant conditioning or scale arithmetic dominates, discriminated
by exact stored-input high-precision references and power-of-two scaling pairs.
Positive expressions alone do not establish relative floating error.

Freeze the manifest before execution. Fixed valid analytic fixtures, exact
collinear/origin/nonfinite guards, dyadic eps2^-10/30/60 controls, and scales
2^-20/2^20/2^-300/2^300. Compare to 160-digit stored-input reference, analytic
unit/orthogonal checks, and direct binary64 endpoint subtraction. Relative
error <=1e-10 is a diagnostic threshold, not a bound. Unsupported finite
collinear fixtures remain unresolved, not failed mathematics or dead methods.

All numerical work uses one thread, 2GB, secondary review lock <=20CPU. Save
once and refuse existing output. Count every explicitly named scalar arithmetic,
comparison/Boolean, abs/sqrt/isfinite operation through Ledger; API type/shape,
loop/index, allocation, Python dispatch, library implementation and MP reference
cost remain separate. Finiteness checks do not certify intermediate rounding.

Future 4D Duffy to 3D outer integration requires a separate frozen gate. In
real arithmetic an exact-inner/GLn hybrid preserves polynomial degree N=n-2
exactness and the same ideal positive-rule bound; this phase makes no such
application or trained accuracy/cost claim.
