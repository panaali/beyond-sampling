# T56: real matrix-free cut contractions

## Question
Can the exact-root-lune projection retain its accuracy while avoiding the
measured dense Gram construction cost?

## Method
Keep the SAME N8/M4 basis span,81 real degrees of freedom, Q64 sine-mapped
radial quadrature and exact half-order support. Represent the angular basis
as1,sqrt2 cos(k theta),sqrt2 sin(k theta), theta=2pi(phi-phi0)/L. Let m_k be
the existing exact angular arc integral of exp(i k theta). Then R00=m0,
R0,cosk=sqrt2 Re m_k, R0,sink=sqrt2 Im m_k,
Rcosk,cosl=Re(m_(k-l)+m_(k+l)),
Rsink,sinl=Re(m_(k-l)-m_(k+l)),
Rcosk,sinl=Im(m_(l+k)+m_(l-k)); remaining entries by symmetry.

At each radial node form F=P C, G=R F, then accumulate P^T diag(w) G.
No complete81x81 cut matrix is constructed. Real b coefficients correspond
to complex a_k=(b_cos-i b_sin)/sqrt2 and a_-k=conj(a_k). The root constant
has coefficient1 and whole-sphere mass1/12. Axial complements use c-cut(c).
Pure-azimuth cuts use exact radial orthogonality, avoiding quadrature.
Inherited arc/basis/topology primitives and geometry are read-only and
source-hashed; original dense engine remains unchanged. A local adapter
uses its same branching sequence with only the cut-space implementation
substituted in a private module instance.

## Frozen alternatives and gates
1. Dense Gram construction dominates: matrix-free application reduces
   ordinary product work and CPU at identical output. Compare cold setup,
   full counts, same modes/order and no cross-model caches.
2. Complex representation/phase mistakes fake the gain: independent real
   angular algebra, nonzero sine coefficients, conjugate mapping, norm,
   full-support identity and complementary cuts must agree with dense cuts.
3. Factorization rearranges floating error near thin arcs: test small axial
   normals, both signs, fixed arbitrary real coefficients and model outputs.
   Require cut max discrepancy<=1e-10 and trained output<=1e-9 before any
   efficiency comparison or ensemble expansion.
4. Elementary functions or branching dominate: reduced products need not
   mean lower CPU or lower total FLOPs. Keep opaque named kernels explicit.
   If equivalence/cost gate passes, repeat the SAME17 frozen cases for parity
   and cold-cost comparison, not a new accuracy search.

No degree sweep. A single lower tangential variant is merely a possible
future task; it is not authorized by this first frozen cycle and will not
be selected after accuracy scoring.

## Controls and first pilot
Real/complex norm and cut parity for normals [0,0,1],[.6,-.7,.2],
[1,.1,.7],[-.1,.2,1e-6], the two support walls and their negatives.
Use deterministic coefficients with nonzero sine/cosine modes, plus root.
Then H2T3/H4T3 base, constants/duplicates and positive hidden rescaling.
No new Monte Carlo, model training or external data.

## Status
Protocol frozen before implementation. T55 independently checked the real
angular identities and coefficient mapping.

## Results
Pending; every failure and correction retained.

## Compute accounting
Charge cold quadrature/basis setup, normalization, topology, arc moments,
angular Gram assembly, all three contractions, weights, complements,
geometry and driver work. Q*(2*nrad*nang+nang^2) counts real products ONLY;
additions/scaling/assembly remain separate. No portable FLOP certificate
while elementary functions and rule construction remain opaque. One thread,
2GB, shared locks, <=110CPU/cycle with internal100 deadline.

## Claims
[inferred] This changes representation/contraction, not finite projection.
[speculative] It improves enough cost to make the scoped gain useful.

## Alternatives not ruled out
Lower/local bases, internal gate support, certified quadrature and better
geometry pruning remain open if this exact-equivalence optimization fails.

## Failure modes
Sine sign errors, complex-to-real normalization, floating cancellation,
missed angular wrap, uncharged setup/special functions and unfair warm runs.

## Initial equivalence and cost gate
Primitive parity maximum1.67e-16, including sine modes, both support walls,
complements and small axial normals. Full H2/H4 estimates differ by
5.56e-17/2.23e-16; constant/duplicate ties and1e-16 hidden scaling pass.
The first implementation's counted ordinary arithmetic falls from
74.22M→15.63M (H2) and289.28M→61.05M (H4), but CPU .120→.112 and
.408→.484 respectively. Thus arithmetic improves while the H4 Python-loop
implementation is slower. No ensemble run or sampling-win claim follows.

Preserve this initial pilot. Next bounded implementation-only alternative:
vectorize angular matrix entry assembly using fixed index arrays constructed
inside each Space. Same formula, nodes, basis and arithmetic; count index
setup and test parity again. This addresses Python overhead only, not a new
approximation or degree choice.

## Vectorized equivalence result
Fixed angular entry assembly is bit-identical on all12 primitive normals.
Base H2/H4 parity remains5.56e-17/2.23e-16. Same-session cold CPU is
.1040→.0841 and.3959→.3360 seconds, with unchanged ordinary arithmetic
counts (new integer index setup separately recorded). This passes the
frozen equivalence/cost gate. Repeat the identical17 prior cases for parity
and cold comparison; no coefficient, degree, node or truth selection.

## Frozen17-case comparison
All prior cases pass numerical equivalence (max1.11e-16 H2,3.33e-16 H4).
These are the same H2 base+8 absolute.05 copies RNG47001 and H4 seed0's
first8 absolute.05 copies RNG0; array hashes and references are preserved.

| Family | Dense mean CPU | Real mean CPU | Dense counted arithmetic | Real counted arithmetic |
| --- | ---: | ---: | ---: | ---: |
| H2 (9) | .0986325 | .0786669 | 74.722M | 15.739M |
| H4 (8) | .352435 | .289961 | 271.764M | 57.370M |

[measured] Cold runtime drops about20.2%/17.7%; ordinary arithmetic about
4.75x/4.74x. CPU ratios are one same-session diagnostic, not timing CIs.
RMSE stays .001210658/.001268277: this is no new approximation improvement.
Squared-error-times-CPU decreases from1.4422e-7→1.1572e-7 (H2) and
5.6869e-7→4.6963e-7 (H4), only relative to this prior dense implementation.
The initial H4 slowdown is preserved in pilot.json, and vectorization is
separately documented; no favorable run was silently substituted.

Each contraction product is an unambiguous scalar-multiplication floor.
It drops from31.3–32.3M to5.15–5.30M for H2, and106.9–123.5M to
17.57–20.30M for H4. With the prior conventional forward count excluding
input generation, mean conditional error²-cost ratios at that FLOOR change
3.994→.6564 (H2) and20.080→3.3003 (H4). A floor below1 is inconclusive,
not a sampling win; additions, arc work, setup, geometry and special
functions still cost work. Four of eight H4 cases remain above1 even at
this reduced floor. This comparison is conditional on the stated forward
convention, not a universal impossibility theorem.

## Reproduction
Run controls.py, pilot.py, vectorization.py, ensemble.py, then cost_floor.py
using the project Python with one thread, caps and the shared lock. Initial
scalar assembly remains the default parameter for exact historical replay;
vectorized=True explicitly selects the tested fast variant. Rule/basis
construction occurs inside every estimate. No model-dependent cache is
reused across calls. Operator-source and unchanged-driver hashes accompany
results; dense canonical files are untouched.
