# T102: real even-arc weighted-tail recurrence

## Question
Can a batched real recurrence reduce the cost of T55's existing analytic two-tail mass primitive without weakening its checks? Claim pushed 9349c69e. This is not a new outer-angle estimator: T47/T51 already exposed that integration's gate/rank bias.

## Method
For f=a cos(s)+b sin(s), g=f', r²=a²+b², differentiation gives (f^(n−1)g)'=(n−1)r² f^(n−2)−n f^n. Starting I0=theta, compute all even arc integrals through degree6 together. Endpoint powers advance by squares. Insert the chain into the unchanged spherical polygon boundary recurrence for odd degrees1,3,5,7 (prefix dimensions2,4,6,8). Baseline is the current matching-parity Fourier implementation, not the older all-degree evaluator. Geometry, geometric bound and positive fallback stay separate and charged.

Frozen alternatives before implementation:
1. Complex Fourier expansion dominates arc work; batched real arithmetic gives a useful primitive speedup. Test same nine polygons from T55 check_parity.py, all four prefix dimensions, repeated fixed300 calls per method on its asymmetric polygon.
2. Endpoint cancellation causes unacceptable arc errors. Test 80-digit positive arc integrals on a,b=(1,0),(0,1),(1,-1),(1,1),(1e−12,1), theta=1e−12,1e−6,.3,pi/2,pi−1e−8, scales1e−6,1,1e6, degrees0/2/4/6. Retain raw failures. A stable endpoint-difference recurrence and explicit positive quadrature fallback are predeclared rescues, not silently selected results.
3. Polygon-level cancellation dominates even if individual arcs improve. Preserve existing thin/equator polygons and guard failures; compare an independent positive polygon rule where needed. A fair trained-prefix pilot follows only after controls, with its cases frozen separately.

## Status
Protocol frozen; implementation pending. One thread, 2GB, shared locks, <=110CPU per cycle; primitive controls <=20CPU review lane. No sampled model inputs or new Monte Carlo.

## Results
Raw controls (`python m16/approaches/real_arc_tail/controls.py`) took4.722CPU:300 arc cases,20 negative real and14 negative Fourier values, with huge relative errors near vanishing integrands. Both variants reject5/36 polygons. Raw benchmark includes redundant construction of the common nine-polygon fixture on each timed call; preserve it, but use a separately reported cached-fixture benchmark for a kernel comparison. No winner selected.

### Rescue protocol frozen after raw controls
Test stable endpoint differences as a separate variant; they remove direct endpoint subtraction but cannot fix recurrence cancellation near a zero integrand. Also test an exact positive centered formula, a concrete alternative to positive quadrature: set A=f(theta/2), B=f'(theta/2), z=sin²(theta/2). For even n and 0<=theta<=pi,
I_n=sum over even j of binom(n,j)*A^(n−j)*B^j*Beta_z((j+1)/2,(n−j+1)/2).
Odd powers of sin integrate to zero on the centered symmetric interval; all retained terms are nonnegative. Evaluate incomplete beta directly through scipy.special, preserving underflow and floating special-function qualifications. Run the same frozen arc/polygon cases and cached-fixture300-call benchmark. This is not a directed arithmetic certificate. Compare full positive and stable variants, not only a post-hoc winner. No trained pilot until this result is recorded.

Rescue result (`python m16/approaches/real_arc_tail/rescue.py`):0.938CPU, stable recurrence still16 negative arcs. Positive beta has none, but initially reported worst relative error1.257e−8. Inspection after endpoint repair showed this was an oracle defect: the unnormalized80-digit quadrature stopped on sin(s)^6 at theta1e−12. Its saved value1.42857141061152056e−85 disagrees with the elementary theta^7/7 asymptotic; the positive formula gives1.4285714285714283e−85. An initial attribution to near-pi rounding was incorrect and is corrected here; near-pi rounding remains a real separate analytic risk. All four polygon variants reject the same5/36 cases. Cached-fixture benchmark records all variants; positive k4/6/8 CPU.0477/.0531/.0647 vs Fourier.0657/.0896/.1248, with k2 timing noise/overhead retained. No generic speed claim.

Next endpoint repair frozen: for theta>pi/2 use complete beta minus the complementary incomplete beta at cos²(theta/2), with swapped parameters. For smaller theta use B_z=sin(theta/2)^(j+1)/p * hyp2f1(p,1−q,p+1,z), p=(j+1)/2; this avoids squaring underflow in the j0 term. Add theta1e−200 and nextafter(pi,0) to the independent positive-integral controls, preserving previous results. Count hypergeometric calls as opaque work. This fixes represented small complements, not arbitrary scale overflow or a certified special-function implementation.

Endpoint result0.967CPU: extra representable controls agree within5.33e−16;3 true underflow cases retained,5 polygon fallback cases retained. Original300 references now require correction: map s=theta*u and normalize the integrand by amplitude=|a|+|b|min(theta,1) before100-digit integration; multiply theta*amplitude^n outside. Recompute references for every variant identically, preserving all old artifacts. The raw/stable negatives already disprove their general positivity independent of the flawed reference.

Normalized oracle repair4.415CPU over332 controls: endpoint formula max representable relative error1.210e−15; original positive beta fails tiny constant-direction arcs by underflow (relative1). Raw/Fourier/stable remain severe cancellation failures. Three true float-underflow cases are explicitly excluded from relative-accuracy maximum, retained in data. Command `python m16/approaches/real_arc_tail/reference_repair.py`.

### Frozen full conditional-kernel pilot
Use public seed0 H2,T=4/6/8/10 checkpoints, no noisy fitting. For each T use normalized prefix proportional to alternating signs with magnitudes1,...,T−2, and its negative (two fixed directions). Evaluate complete old and endpoint kernels from the same unmodified T55 geometry, preserving all stats, input model loading and prefix setup time. Then repeat each conditional query three times per method for timing; source geometry is rebuilt each time, no cached answer. Record probability and partition differences, fallback counts/error bounds and all opaque special-function counts. Expected alternative: geometry cost dominates and total speed benefit is small even when arc work falls. Any explicit failure is retained; no other prefix is substituted. No new MC, full outer integration or estimator-win claim.

## Compute accounting
Charge normalization/edge geometry, all arc chains, endpoint trig/powers, polygon recurrence, guards and fallback construction/query work. Report named operation categories and CPU including controls separately. Do not call a partial scalar proxy a portable full FLOP audit. Baseline Fourier term count is vertices*(1+3+...+degree), matching parity only.

## Claims
[inferred] Real recurrence is algebraically identical in exact arithmetic (T55 independent derivative check). [speculative] It may lower existing conditional-kernel cost. No full accuracy, topology certification, sampling win or ladder claim.

## Alternatives not ruled out
Stable endpoint differences, long-double arithmetic, positive fallback, vectorized Fourier and compiled implementations remain open. Outer quadrature and prefix/rank coupling are unchanged.

## Failure modes
Cancellation in endpoint subtraction or polygon boundary sums; tiny arcs; under/overflow at extreme scales; rounded geometry and incomplete topology. Passing positivity bounds is not an accuracy certificate. Existing T55 fallback bounds cover integration only.
