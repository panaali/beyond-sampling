# T175 first analytic gate

Claim2666600d. This is a mean-only real-arithmetic route, not a covariance,
step4, final-accuracy or full-compute result. No model/reference/MC input replay.

For independent R~Rayleigh(1), Z~N(0,1), E ReLU(aR+bZ) equals
[|b|+a(pi/2+atan(a/|b|))]/sqrt(2pi), with b=0 limit
max(a,0)*sqrt(pi/2). The initial implementation is fixed-precision MP control
code, not an optimized binary64 kernel. Negative-a cancellation is deliberately
exposed against independent positive polar-angle integration. A later stable
series/binary64 implementation would need a new frozen gate and costs.

Alternative exact first moment: for distinct actual facets of a polyhedral
cone C, Gaussian integration by parts gives
E[G 1_C]=(1/sqrt(2pi))*sum_i n_i P(N(0,I) restricted to n_i-perp lies on facet i),
where n_i are inward unit normals. Add v.G>=0 to the two h2 angular-sector
halfspaces, then dot with v. Distinct nonparallel normals are handled with
exact stored-dyadic Gram algebra. Positive-proportional duplicates merge;
zero constraints vanish; opposite constraints and zero output return zero.
Redundant planes of lower-dimensional contact have exact zero facet measure.

With at most three distinct normals, each facet probability is full plane,
one halfplane, or the intersection of two halfplanes. For the latter, let p,q
be projected squared norm numerators and b their projected dot numerator.
Its wedge angle is atan2(sqrt(pq-b²),-b), so probability is angle/(2pi).
The radicand is computed exactly as a Fraction, not clipped from a floating
correlation. Signed v dot n terms can still cancel. This is not a full outward
floating certificate or a generic higher-dimensional cone solver.

Freeze exact dyadic analytic/domain fixtures before execution. Compare facet
outputs at160/200 digits and independent angular radial integration at120
digits, splitting any b=0 output zero crossings. Reference differences use
fixed absolute1e-90 plus relative1e-80 tolerance for nonzero values; zero targets
require absolute1e-100. All finite precision failures remain explicit. No
precision increase, fixture choice or quadrature rule selection after outcomes.

Two competing explanations are a wrong facet/domain multiplicity versus signed
sum cancellation; exact duplicate/opposite/redundant controls separate them.
A second comparison tests loss of within-sector radial structure: radial-angle
integration is independent of Gaussian boundary integration, so disagreement
is an implementation/reference limitation rather than evidence closure improves.

One thread2GB, secondary review lock20CPU. Append each attempt and result before
assertions and refuse existing outputs. Exact arithmetic counts and retained
reduced numerator/denominator bits are setup evidence only; MP integration,
conversion/loops/gcd/runtime and full native costs remain separate. No source
T69 estimate is modified, and its large covariance/downstream errors persist.
