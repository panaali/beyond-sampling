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
Protocol frozen; proof/implementation next.

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
