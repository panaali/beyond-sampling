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
Pending. Reproduction commands will accompany each result. No winner selected.

## Compute accounting
Charge normalization/edge geometry, all arc chains, endpoint trig/powers, polygon recurrence, guards and fallback construction/query work. Report named operation categories and CPU including controls separately. Do not call a partial scalar proxy a portable full FLOP audit. Baseline Fourier term count is vertices*(1+3+...+degree), matching parity only.

## Claims
[inferred] Real recurrence is algebraically identical in exact arithmetic (T55 independent derivative check). [speculative] It may lower existing conditional-kernel cost. No full accuracy, topology certification, sampling win or ladder claim.

## Alternatives not ruled out
Stable endpoint differences, long-double arithmetic, positive fallback, vectorized Fourier and compiled implementations remain open. Outer quadrature and prefix/rank coupling are unchanged.

## Failure modes
Cancellation in endpoint subtraction or polygon boundary sums; tiny arcs; under/overflow at extreme scales; rounded geometry and incomplete topology. Passing positivity bounds is not an accuracy certificate. Existing T55 fallback bounds cover integration only.
