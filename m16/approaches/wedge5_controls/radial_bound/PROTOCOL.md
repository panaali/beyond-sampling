# T11583 fixed radial-bound continuation

Claim36828ec0 precedes implementation. No mass/model/MC evaluations: reuse the11 exact stored dyadic ray matrices and GL6/10/16 numerical outputs from the independently reviewed original66 controls.

For q=y'Gy on the4-simplex, b=max diagonal is an upper bound. Exact nonnegative pair products imply a=min diagonal/5. Otherwise enumerate all31 nonempty supports, solve each exact equality minimizer, retain nonnegative candidates, and validate the minimizing full-vector KKT inequalities/complementarity. Original dyadic geometry remains unchanged. Determinant, Gram and solve construction use separately counted Fraction operations; rational bit complexity and process CPU remain distinct from numeric kernel work.

With nu=5/2, c=(a+b)/2, r=(b-a)/(b+a), the binomial absolute tail after degree N is E=a^(-5/2)-c^(-5/2)sum_(k=0)^N((5/2)_k/k!)r^k. Polynomial total degree2N in barycentric coordinates becomes maximum degree2N+3 in the four Duffy variables including the Jacobian. For N=n-2 GLn integrates that polynomial exactly. Exact and positive quadrature functionals both have simplex weight1/24, giving |det R|E/(32pi²).

Use unchanged pinned Directed50 implementation, expanded exact-input radial interval endpoints, outward reciprocal/sqrt/powers and outward subtraction of the partial series from the full series. A negative lower tail endpoint is intersected with known tail>=0; a negative upper endpoint is an explicit failure. The returned lower formula endpoint is NOT a lower bound on actual quadrature error. Precision/order/target remain50,6/10/16,1e-10. Preserve loose bounds and finite precision failures. Cold Machin pi setup and every directed operation remain recorded, with exact coefficient arithmetic separately recorded. MP120 formula checks are separate non-directed diagnostics, not the certificate.

Freeze three range controls: constant q=1, a=1/2,b=1, and a=1e-30,b=1, determinant1, at all three orders. No targeting or adaptive rescue after outcomes. Refuse existing results and save partial failure on exceptions. Source gate precedes execution;20CPU review lock, one thread,2GB.

This certifies only ideal real-arithmetic GL truncation for the stored-ray geometry. It does not bound the source's determinant, normalization, nodes, weights, kernel arithmetic, log-rule error or trained-model errors. An unresolved bound is not proof of quadrature failure or exhaustion of other rules/representations. Positive edge splitting is independently registered T11586.
