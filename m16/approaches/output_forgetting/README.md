# T105: output forgetting with the full rank law

## Question
Can a shorter input event make the network's output label independent of its earlier hidden state, even when hidden amplitude retains memory? Claim pushed/rechecked f2ca9e8b. T50 independently proved that public H2T10 needs at least five inputs for a positive-probability uniform hidden-zero reset; output forgetting is a different property.

## Method
For Wi=(-alpha,beta), Wh=[[a,-b],[c,-d]], positive parameters, bc>ad and a²<bc, two positive inputs force hidden onto the second axis. A third positive xC gives h2 in[0,beta*xC]; a fourth xD>d*xC gives h2>0 and h1=0 for every initial state. Thus on event E={xA,xB,xC,xD>0, xD>d*xC}, the output is the first-index maximum of Wo[:,1]. The event probability is (1/4)*(1/4−atan(d)/(2pi)). Exact rational coefficient checks and readout ties precede use of this theorem.

The earlier m=T−4 inputs still affect target rank. For u>0 write F=Phi(u), Q=1−F, G=F−1/2. Given target xD=u, the joint second-rank density is phi(u) times the coefficient of z¹ in (F+Qz)^m (G+Qz)^2 [G+(Phi(u/d)−F)z], for0<d<=1. Given target xC=u, replace the final factor by [(F−Phi(d*u))+Qz]. For a target in the earlier prefix, use (F+Qz)^(m−1)(G+Qz)^2(g0+g1*z); for target xA or xB use (F+Qz)^m(G+Qz)(g0+g1*z). Here g0 is the pair-event mass with both xC,xD<=u, and g1 with exactly one above u:
g0=integral_0^u phi(v)[F−Phi(d*v)]dv;
g1=G*Q+integral_u^(u/d) phi(v)[F−Phi(d*v)]dv.
All integrands are nonnegative; transform v to Gaussian CDF coordinates. Target integration likewise uses F in[1/2,1]. Handle m0 and d0 separately without0*infinity. The d0 case is a polynomial control: E simply means all four suffix values are positive.

## Status
Protocol frozen before implementation. This produces a partial-event accuracy contribution and its exact event coverage, not a full accuracy estimator. One thread, <=2GB, <=110CPU cycles with shared locks; short controls may use review lock.

## Frozen controls and experiment
1. Exact parameter and readout-column checks on public H2T10 seed0. Verify the uniform prediction using T50's exact projective cells at three fixed event words(1,1,1,1),(1,2,3,2),(3,1,1/2,1), both model hidden coordinates and readout first-index ties. Zero/duplicate readout controls retain the positive-axis condition; an outside-event word is a falsifier candidate, not silently included.
2. Model-free rank controls at T4/T6/T10 and d0,1/4,1/2,1: nonnegative joint masses, exact all-target partition m*A+2*B+C+D=P(E), prefix and free-positive exchangeability, d0 exact polynomial antiderivatives. Independent pair masses use the Owen-T identity H(u)=.5*(Phi(u)−.5)+atan(d)/(2pi)−T(u,d), with g0=F*(F−.5)−H(u), g1=G*Q+F*(Phi(u/d)−F)−H(u/d)+H(u). Preserve cancellation failures; no negative clipping.
3. Fixed positive quadrature orders outer/inner(32,32),(64,32),(32,64),(64,64), all reported with full node/CDF/rule setup cost. Frozen same rules on actual d. Independent adaptive scalar integration of the analytic pair formula checks the final event contribution, with numerical error estimates explicitly not certified. No new random-input validation or model selection.

Alternatives: output forgetting covers a material event with a cheap rank-aware contribution; or its event coverage/conditional correctness is too small to help a full estimator. Missing prefix ranks can produce deceptively plausible results; the all-target partition and d0 exact controls are the primary falsifiers. Numerical CDF/Owen cancellation is separate from the theorem and must be preserved/repaired explicitly.

## Results
Pending. Exact event coverage is not estimated accuracy. Unresolved complement mass is retained.

## Compute accounting
Charge exact coefficient/readout setup, every rank polynomial/CDF/inverse-CDF/Owen call, quadrature-rule generation and evaluation, and projective validation separately. Special-function calls and Fraction operations are not silently priced as ordinary FLOPs. No full model-input samples or uncharged model-specific advice.

## Claims
[inferred] The four-input event yields a constant output label under the explicit weight inequalities. [speculative] Exact event contributions could reduce effective history cost. No solved rung, global memoryless model or full accuracy claim.

## Alternatives not ruled out
Other sign patterns and output cells, unions of events, sharper event cones, direct certified scalar integration, and the effect on a complete estimator remain open. No hidden-reset minimum is transferred to output forgetting.

## Failure modes
Leaving positive-axis support; first-index readout ties; omitted earlier ranks; zero/negative quadrature artifacts; insufficient tail/outer convergence; overlaps if event families are combined. This initial event uses the final four positions only, so no overlapping-window approximation is made.
