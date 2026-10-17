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
Model-free controls (`python m16/approaches/output_forgetting/controls.py`) take.1481CPU. Exact d0 polynomial controls agree within2.09e−17; all computed masses are nonnegative. Fixed positive quadrature still has partition defects up to9.32e−8 and pair-mass differences up to4.18e−8 versus the Owen identity. These are retained numerical integration errors, not silently normalized away.

Public T10 pilot (`python m16/approaches/output_forgetting/pilot.py`) takes.8726CPU including torch/load/setup. All three exact event-word output certificates and the zero/duplicate readout controls pass. The outside-event word fails uniformity as expected. Axis winner is index9 (position10). Event probability is.05536152383968549; all four frozen quadrature combinations give joint correct mass.00922267905888864..866. Independent adaptive scalar integration using the analytic Owen pair law gives.009222679058888648, with estimated vector integration error9.20e−14,483 adaptive nodes and966 Owen calls in.01079CPU. This reference's reported error is numerical, not a certificate. Other target classes retain finite positive-rule partition error up to1.303e−7; matching the selected class does not establish convergence of every class.

This contributes about.9223 percentage points to total accuracy on5.536% of input mass. It is not the total model accuracy; the complement mass.9446384761603145 is unresolved. The rank-aware conditional correctness given this event is about16.66%, so replacing earlier ranks with an unconditional1/T guess would change the result materially. The event proof applies to arbitrary initial states, while all earlier values remain in the target rank law.

## Compute accounting
Charge exact coefficient/readout setup, every rank polynomial/CDF/inverse-CDF/Owen call, quadrature-rule generation and evaluation, and projective validation separately. Special-function calls and Fraction operations are not silently priced as ordinary FLOPs. No full model-input samples or uncharged model-specific advice.

## Claims
[inferred] The four-input event yields a constant output label under the explicit weight inequalities. [speculative] Exact event contributions could reduce effective history cost. No solved rung, global memoryless model or full accuracy claim.

## Alternatives not ruled out
Other sign patterns and output cells, unions of events, sharper event cones, direct certified scalar integration, and the effect on a complete estimator remain open. No hidden-reset minimum is transferred to output forgetting.

## Failure modes
Leaving positive-axis support; first-index readout ties; omitted earlier ranks; zero/negative quadrature artifacts; insufficient tail/outer convergence; overlaps if event families are combined. This initial event uses the final four positions only, so no overlapping-window approximation is made.

## Frozen follow-up: remove inverse-CDF endpoint regularity
The retained all-target partition defect could come from endpoint nonsmoothness of the inverse-CDF coordinates, or from an incorrect rank formula; the analytic pair reference and all-target partition discriminate these. Before further runs, freeze a direct positive Gaussian rule: target u on [0,8], paired v on [0,u] or [u,min(u/d,8)], positive density factors and survival-CDF differences. Keep the G*Q term exact. Each class loses at most 2*Phi(-8) by the union bound for the target and the single integrated pair coordinate; T times this is a conservative partition omission bound. This is a tail bound only, not a quadrature or floating-roundoff bound. Compare fixed outer/inner orders (64,32),(128,32),(64,64),(128,64) on T4/6/10, d0,1/4,1/2,1 and public T10's d. Adaptive Owen pair integration is diagnostic; no renormalization or negative clipping. Count all rules, nodes and elementary/special functions and preserve any nonconvergence. This bounded discriminator is part of T105, not a new model or event selection.

The frozen direct rule (`python m16/approaches/output_forgetting/direct_gaussian.py`, `direct_results.json`) completes all52 comparisons in.3891CPU. Maximum all-target partition defect is4.51e-16, maximum class difference from adaptive Owen integration1.16e-16. The lowest rule already agrees to rounding in this fixed panel; the old inverse-CDF defect is preserved and is evidence of a numerical coordinate issue, not an incorrect rank law. The public model's four class masses are [.003746991404621079, .008079137863453371, .007498620626163730, .009222679058888667], with multiplicities[6,2,1,1]. Its64x32 rule costs.00440CPU,96 rule-generation nodes,64 outer nodes,4096 inner nodes,8576 CDF values and4160 exponentials. These are a work inventory, not audited portable FLOPs. Each target-class tail loss is bounded by1.245e-15, with conservative ten-class partition bound1.245e-14; those bounds exclude numerical quadrature and floating roundoff. No empirical exactness or global accuracy claim follows.

**Current status:** bounded implementation/pilot complete, independent reviewer requested. A complete estimator still needs the complement event and audited cost. All-positive event, output and rank statements are conditional on the explicit exact weight inequalities. Retaining arbitrary hidden initial states and integrating all preceding ranks is essential. Broad family alternatives remain open.
