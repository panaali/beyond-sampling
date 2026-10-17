# Untested first-sign mixture: a distinct retained alternative

This note is mathematical design only, not an implemented estimator or a
new task claim. Root has been asked to register it as a future T76 phase.
It does not change the frozen deterministic-mean pilot or its classification.

Condition on S=sign(X0), which has probability1/2 each. For a two-step prefix,
the conditional means are exactly

mu_s = (hypot(c_s, Wi) + c_s) / sqrt(2*pi),
c_s = Wh relu(s*Wi).

Their average equals the unconditional exact2 mean already tested. A mixture
can retain a gate-history bit that a single deterministic vector discards.
For longer prefixes, any further mean closure remains approximate.

For fixed suffix values y,x and m omitted prefix inputs, let n=m-1,
F(v)=Phi(v), Gplus(v)=max(0,2F(v)-1), Gminus(v)=min(1,2F(v)). The conditioned
first prefix has CDF G; the other n prefixes remain independent standard
normals. Define A(v)=G(v)F(v)^n, the probability no prefix exceeds v, and
B(v)=(1-G(v))F(v)^n+nG(v)(1-F(v))F(v)^(n-1), the probability exactly one does.
The term with n must be zero when n=0, without evaluating a negative power.

The first suffix is true second-largest with probability A(y) if x>y and
B(y) if x<y. The last suffix has probability A(x) if y>x and B(x) if y<x.
These replace the iid-prefix powers in the deterministic surrogate.

Crucially, earlier labels are no longer exchangeable: label0 has the
conditioned sign, the other n are exchangeable. With L=min(x,y), U=max(x,y),
its probability of being second-largest is

D(L,U) + E(U,infinity),
D(a,b) = integral_a^b G'(v) F(v)^n dv,
E(a,b) = integral_a^b G'(v) n(1-F(v))F(v)^(n-1) dv.

Here G'(v)=2phi(v) on the chosen sign half-line and zero elsewhere. Clip
each interval to that half-line; the primitives are respectively
2F^(n+1)/(n+1) and2[F^n-nF^(n+1)/(n+1)]. For n=0 the second integral is zero.
Each ordinary omitted label gets (1-Psuffix_first-Psuffix_last-Pprefix0)/n
when n>0. Reusing the old equal1/m pooling would be a mathematical error.

The two history surrogates are still suffix-only once S is fixed, so these
rank laws exactly score the mixture surrogate. They do not make it equal to
the actual recurrent network. The x-integrals become piecewise polynomials
in F(x), with additional switches at x=0 and x=y; affine arrangement and
Gaussian-landmark machinery could be reused only after a new frozen proof,
controls, cost ledger and claim. No performance result follows from this note.
