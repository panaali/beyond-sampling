# T78 directed positive Chebyshev tails

## Question
Can the existing T6 inverse-cube simplex error planner evaluate its analytic tail with a directed arithmetic enclosure? Claim cf01e28 was pushed before this protocol. This addresses arithmetic of bounds, conditional on valid radial endpoints/determinants and geometry, not full topology certification.

## Method
For q=m+delta*x on[a,b], let s=sqrt(ab), r=(b-a)/(a+b+2s), and z=m/s-1=(b-a)^2/[2s(a+b+2s)]. The k>=1 Chebyshev coefficient magnitude of q^-nu is r^k*s^-nu*P_nu(k,z). P_1=2 and
P_(nu+1)=((k+nu+nu*z)*P_nu+(z^2+2z)*d_z P_nu)/nu.
All polynomial coefficients are nonnegative rational numbers, so fixed small integer nu gives a positive tail sum. Expand k=N+j, N=degree+1; geometric moments of j use positive Eulerian polynomials divided by powers of(1-r). This removes cancellation and gives an exact finite expression for every positive integer nu.

Implement rational coefficient generation and directed Decimal interval arithmetic. Input floats are converted exactly with Decimal.from_float; sqrt is correctly rounded then expanded by next_minus/next_plus. Addition/multiplication/division/integer powers and all sums use separate directed contexts. Derive pi bounds from the alternating Machin identity (16 atan(1/5)-4 atan(1/239)), not an unchecked decimal literal. Convert bounds to float only with an explicit outward step or retain Decimal endpoints. The constant interval a=b has tail exactlyzero. Reject invalid bounds/degrees rather than returning a plausible finite value.

For d=2nu and a determinant enclosure D, the positive-quadrature error is D*(nu-1)!/[pi^nu*(2nu-1)!] times the coefficient tail. This encloses the analytic quadrature bound given the inputs. Omission and inner-integration bounds must remain separate and are added directionally; no ordinary computed radial endpoint is promoted to a valid enclosure.

## Frozen alternatives and falsifiers
1. Ordinary bound evaluation may already be accurate enough versus a cancellation/underflow defect. Compare the old q^-3 float formula to the directed result on fixed narrow intervals and large degrees; preserve any undershoot without claiming it changes old estimator conclusions.
2. The positive recurrence could have a missing normalization or shifted tail index. Independently compare nu1 closed geometric sum, nu2/nu3 known explicit formulas, and high-precision coefficient integration or positive Fourier products for nu4/nu5. First omitted coefficient is N=degree+1.
3. Interval dependency could make bounds too loose or expensive. Freeze decimal precision50 then80 as validation, degrees0/3/15/63/255/2047, radial intervals[1/6,1], [0.9,1], [nextafter(1,0),1] and [1e-100,2e-100]. Report widths, containment, operation counts and cold setup separately; no tolerance picked from trained accuracy.
4. An enclosed analytic tail is not an enclosed model accuracy. Preserve input/topology assumptions; first demonstrate concrete use by re-evaluating a fixed small subset of saved T6 planner intervals only if their provenance can be enclosed independently. Otherwise expose the primitive interface and explicitly leave integration blocked on valid inputs.

## Frozen execution
Implement/prove first; <=20CPU fixed primitive controls under review lock, one thread/2GB. No model forwards or new full integration. Record every arithmetic operation and precision, coefficient/pi setup, failures and source hashes. No hidden arbitrary-precision operations counted as unit FLOPs; counts are named precision-dependent kernels. T55 independently reviews recurrence and endpoints.

## Status
Bounded primitive and concrete16-cone use complete; independent implementation review pending.

## Results
Pending.

## Compute accounting
Numerical cycles <=110CPU under compute lock, review controls <=20CPU under review lock; ulimit -v2500000. Directed Decimal arithmetic and rational setup have explicit event counts and CPU, not portable FLOP claims. No per-model truth or fitted constants.

## Claims
[inferred] The positive recurrence follows from -1/nu times the m derivative at fixed delta. [speculative] Directed evaluation is inexpensive enough to use in current T6 planning. No numerical enclosure or estimator success yet.

## Alternatives not ruled out
Binary directed arithmetic, exact algebraic bounds for sqrt, tighter error allocation, and higher-even-dimensional uses remain open. Generic noninteger tails are outside this phase.

## Failure modes
Incorrect input enclosures, Decimal exponent underflow/overflow, pi-series endpoint sign, tail-index mistakes, integer recurrence overflow, accidental float conversion and assuming directed tail arithmetic fixes floating geometry. Reject arithmetic exceptions and require strictly positive lower endpoints.

## Algebra before controls
Differentiation at fixed delta gives r'/r=-1/s, s'=1+z and z'=-(z²+2z)/s. Thus -d_m/nu maps coefficient magnitude r^k*s^-nu*P_nu to the stated recurrence. In coefficient form, a monomial c*k^j*z^l contributes c/nu to(j+1,l), c*(nu+2l)/nu to(j,l), and c*(nu+l)/nu to(j,l+1), all nonnegative. The alternating Chebyshev sign(-1)^k is unchanged. For N>=1, sum_(k=N)^infinity k^j r^k=r^N sum_(l=0)^j binomial(j,l)N^(j-l) sum_(v=0)^infinity v^l r^v. The last sum has Eulerian numerator for l>=1, and1/(1-r) for l0. The error factor includes twice the polynomial remainder and simplex volume1/(2nu-1)!, yielding (nu-1)!/[pi^nu(2nu-1)!].

## Retained harness overflow
The first fixed run stopped when its test harness unconditionally requested binary floats for a valid Decimal tail larger than binary64 maximum (small radial interval, high inverse power). The primitive correctly raised OverflowError. The original failure is retained; the harness now requires that rejection only when the independently computed truth exceeds binary64 maximum. It still checks outward conversion for representable/underflowing tails. No arithmetic implementation or fixed cases changed.

## Primitive results and concrete-use protocol
All240 predeclared tail cases at precisions50/80 enclose an independent160-digit derivative of the inverse-linear closed tail. Maximum relative interval width5.389e-46;10 simplex constants and directed sums also enclose independent references. Total.156CPU/38MB, excluding imports. Twelve of24 old float q^-3 values undershoot the mathematical tail on this adversarial panel, including underflow; this does not establish any changed trained-estimator conclusion. The original float formula/results stay unchanged.

Freeze the concrete-use test now: first16 rays from the existing cycle14_cold_rays.npz in storage order, Jacobi orders4/8/12/16, no selection by accuracy. Preserve them as an exact binary-float fixture with source hash. Compute every Gram entry, a supporting-plane lower bound and determinant using exact Fraction arithmetic on the stored float coordinates. Let w be the ray sum; lower=max(min Gram, min(w dot v)^2/(w dot w)) when all projections are nonnegative, and upper=max squared vertex norms. These bound all convex combinations by positivity/Cauchy and convexity. Convert lower down and upper up; pass an exact determinant enclosure into the new tail. Zero/negative lower bounds explicitly fail this phase rather than being patched by the unverified L1 assumption. Compare mass-bound intervals against an independent160-digit q^-3 formula on the same rounded radial interval, plus exact-rational geometry checks. Sum all16 bounds directionally. This certifies bound arithmetic for the stored simplices, conditional on their geometry representing the original event; it is not a new accuracy integral or model enclosure. Charge rational endpoint/determinant setup separately, preserving earlier floating plans.

## Concrete T6 subset result
All64 predeclared per-cone/order bounds and all four directed16-cone sums contain independent160-digit explicit inverse-cube formulas on the exact same radial intervals/determinants. Exact Fraction geometry produces valid lower/upper radial bounds and determinant enclosures for every stored simplex; no normalized-ray assumption or ad hoc1e-12 padding is used. Independent multiprecision determinants agree with the exact rational determinants. The durable16-cone fixture records the original cache hash and storage indices.

Total.112CPU/57MB, including fixture acquisition and independent validation, excluding interpreter imports. Instrumented rational setup totals.012CPU and bound calls.052CPU; short per-call timings are clock-quantized and not used for a speed claim. Named rational and directed-decimal operation counts, precision and maximum elimination integer bit lengths are preserved. These are not portable FLOPs; Fraction operations include size-dependent integer arithmetic and implementation overhead. No model accuracy integral was rerun.

The new primitive corrects a real arithmetic-certification gap in the old floating bound, including underflow controls. It does not retrospectively make earlier geometry/planner outputs certified. A full accuracy certificate still needs valid geometry, integral roundoff/inner bounds, omission and outward combined sums. No estimate, tolerance, candidate order or existing result was changed.

Reproduce under the review lock: `python check.py` and `python t6_subset.py`. The latter uses the tracked fixed fixture after its first acquisition. Public API: `Directed(precision).tail(a,b,nu,degree)` and `.simplex_bound(a,b,nu,degree,determinant_interval)`; endpoints remain Decimal, optional `.floats()` rounds outwards or rejects overflow.

## Accounting limits
Every directed Decimal arithmetic call is counted. Rational endpoint/determinant work is counted as Fraction add/multiply/divide/negate events, with integer bit sizes, rather than as unit-cost machine operations. Coefficient construction and Machin-pi setup additionally report polynomial/series terms; Python indexing, integer powers/binomial/factorial construction, Fraction normalization and bigint implementation work remain named setup kernels whose elapsed cost is included. The event ledger is deliberately not a complete portable FLOP audit. Pi is rebuilt by each simplex-bound call in this prototype; this overhead is charged and not hidden behind cross-model caching.

## Independent source review and conversion-edge repair
T55 independently confirms the recurrence, coefficient normalization, Eulerian moment sums, determinant elimination and exact-rational radial bounds. It identified an output API edge: a Decimal endpoint just above binary64 maximum rounds to finite maximum before the outward correction; nextafter then becomes infinity after the original finite check. The returned interval remained enclosing, but violated the promised reject-overflow contract. The exact reproduced endpoint and old return are retained in float_max_failure.json. A second finite check after correction now rejects this case; core Decimal arithmetic is unchanged.

## Frozen binary64 cost alternative
The Decimal prototype is a correctness reference; per-call rational pi construction and arbitrary precision can dominate a large planner. Freeze a separate vectorized inverse-cube-only variant, using the same positive closed tail and explicit basic operations. Each nonnegative arithmetic result is expanded one binary64 neighbor outward; sqrt assumes correctly rounded IEEE binary64, as do basic arithmetic and conversion. Integer powers use repeated outward multiplications, never opaque np.power. Pi endpoints come from the directed Decimal/Machin primitive and are charged once per invocation. Positive underflow retains an upper subnormal; non-resolved denominators/nonfinite intermediate expressions return a conservative[0,+inf] interval and diagnostic count. Constant intervals and zero determinants return exactlyzero. No narrowing, tolerance fitting or substitution into old saved planners.

Before timing, compare all existing24 q^-3 edge cases and all64 exact-rational T6 subset bounds with the preserved Decimal intervals and160-digit formulas. The binary interval must contain the independent mathematical value; it need not contain the wider Decimal interval if an endpoint coincides, but report any disagreement. Use scalar degree at a time (0..2^26-1, so N squared is exactly representable), vectorize rays, and record operation/nextafter/sqrt counts plus unresolved/infinite results. Cost phase is the same16 cones/four orders, then a predeclared64-repeat tiling solely to measure overhead; no new model/cone cases or favorable repeats. Report the fixed small corpus and repeated arithmetic workload distinctly. Source geometry and full integration remain unchanged.

## Binary64 alternative result
All24 primitive q^-3 controls and64 T6 subset bounds enclose independent high-precision values. The subset uses outward binary radial endpoints, so its independent formula is re-evaluated on those slightly wider intervals. All original Decimal intervals are also contained for the primitive controls. Maximum subset relative width4.216e-14; underflow controls can have enormously loose relative widths despite tiny absolute bounds. No unresolved/infinite result occurs on the concrete subset. The repeated timing corpus is explicitly only16 distinct cones:64 evaluations take.00467 wall seconds;4096 tiled evaluations take.00737 seconds, including four fresh Machin-pi setups. CPU measurements are quantized (.008/.004 seconds) and are not interpreted as monotone or a portable speed claim. Rational geometry preparation is additional. Full numeric nextafter/arithmetic and setup-kernel events are saved.

Before handoff, freeze four additional output/domain controls for the new extended-bound behavior: constant radial interval, zero determinant, positive denominator underflow (a1e-300,b2e-300), and large product overflow (a1e300,b2e300), all degree3. Require exactzero on the first two, and interval containment or explicit conservative infinity for the last two against a160-digit formula. These test declared failure modes, not model performance or tolerance selection.

The additional domain controls caught a conservative but incorrect API detail: exactzero tail was widened to a subnormal upper endpoint by the later determinant/constant scaling. binary_constant_failure.json preserves it. The final mass result now applies the exact a=b or determinant-zero branch after all scaling; no tolerance or approximate-zero test was added.

After the exact-zero API repair, all24+64 containment controls still pass. The retained final timing is.00438/.00676 wall seconds for64/4096 evaluations; earlier timings remain recorded rather than selecting the faster run. Both extreme product/denominator cases conservatively return[0,+inf], as promised, while constant and zero determinant return[0,0]. Final relative-width metrics use decimal strings because adversarial underflow can make the ratio exceed binary64 range; original extended-JSON Infinity metrics remain historical, not finite accuracy guarantees. See binary_results_final.json and binary_edge_results.json.

## Directed planning-sum completion
T55 Decimal review6d810bb independently checks source/algebra and18 closed-form edge cases. Complete the binary helper with a positive pairwise outward sum so callers need not use an ordinary NumPy reduction on individual upper bounds. Freeze empty/single/16/17-element sums, subnormals and an infinite-upper unresolved input, compared to exact Fraction sums where finite. Then apply the same sum to each existing16-cone/order batch; no new rays or integration. This adds summation accounting and leaves individual bound formulas unchanged.

Pairwise outward sums pass all five finite controls and the infinite-upper control against exact Fraction arithmetic. Each16-cone bound sum encloses the exact sum of its supplied binary endpoints, with30 additions and30 nextafter calls. The empty sum is exactlyzero. The short test CPU timer returnedzero and is recorded as unresolved, not zero computation cost. `check_sum.py`/`sum_results.json` retain the evidence; this addition does not alter previously checked individual bound formulas.
