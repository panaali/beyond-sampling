# Real supported-lune operator: independent review

The operator is an algebraic re-expression of the same81-dimensional
projection and radial quadrature, not a new accuracy approximation. Let
m_k be the normalized cut integral of exp(i*k*theta). In the real basis
[1,sqrt(2)cos(k theta),sqrt(2)sin(k theta)], angular Gram entries are
Rcc(k,l)=Re(m_(k-l)+m_(k+l)), Rss(k,l)=Re(m_(k-l)-m_(k+l)), and
Rcs(k,l)=Im(m_(l+k)+m_(l-k)); constant cross-entries are sqrt(2) times
Re/Im(m_k). Source scalar and vectorized assembly implement these signs.
Real coefficients map to a_k=(bcos-i*bsin)/sqrt(2), a_-k=conj(a_k).

For a literal normalized angular interval[.17,.63], an independent50-digit
positive quadrature directly integrates all25 real basis products. It does
not call the owner's arc geometry or dense Gram implementation. Both real
assemblies agree within5.55e-17. Independent field evaluation at three
points, for three radial coefficient rows, confirms the real/complex map
to4.48e-16. The partial-interval matrix is positive definite in this control.
These checks took .82230 CPU seconds after imports.

Read-only operator_space.py/adapter.py review confirms the contractions
F=P*C, G=R*F, result+=P^T*(weights*G). Their product-event count is
Q*(2*nrad*nang+nang^2); weights and result accumulation are charged
separately. The azimuth-only operator uses nrad*nang^2 products. Complement
and real constant-mode mass c0/12 preserve the earlier lune normalization.
Angular assembly charges5*M^2+2*M operations per radial node. The driver
adds two ordinary arithmetic operations per contraction product event,
which bounds multiply-and-add arithmetic; this is not hardware timing.
Special functions, rule construction and named comparison/index costs
retain their existing explicit qualifications. No portable total certificate
follows from the ordinary-arithmetic subtotal.

Independent raw saved-aggregate arithmetic reproduces all17 case summaries:
H2 mean ordinary arithmetic74.722M to15.739M and CPU .09863 to .07867;
H4 arithmetic271.764M to57.370M and CPU .35244 to .28996. Estimates agree
with the dense operator within1.11e-16/3.33e-16, so existing RMSE is unchanged.
The roughly4.74x subtotal reduction yields only about1.25x/1.22x observed
runtime improvement, not an equal-sized end-to-end gain. The earlier scalar
assembly slowdown remains recorded. Projection bias, full geometry cost,
opaque functions and the original challenge's scaling requirement remain
unresolved. Source parity plus independent angular signs support this finite
implementation improvement, not a new sampling-win or accuracy claim.
