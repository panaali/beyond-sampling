# T141 conditional Gaussian retained-margin coordinate

## Question
Can retaining the exact scalar Gaussian law resolve the T138 same-gate ambiguity, and does mean replacement have a universal bias sign? Claim ec89608 was pushed before implementation.

## Method
Pin T138 exact pair and margin certificate. Set x(lambda)=a+lambda*d, where d=b-a and a is correct. Conditional on the Gaussian input's component orthogonal to d equalling this line's fixed transverse component, lambda has normal mean mu=-(a dot d)/(d dot d) and variance1/(d dot d). Condition further on one of the two predeclared intervals[0,1] and[1/2,1]. T138 independently proves all gates and ranks unchanged, with correctness exactly lambda<c for the rational crossing c, except the measure-zero output tie.

Use the analytic normal-CDF ratio to integrate correctness and the analytic truncated-normal mean for the mean representative. Freeze60/90-digit evaluations to check numerical stability; agreement is not a certified transcendental interval. Independent reviewer uses direct scalar quadrature. Analytic controls: standard symmetric interval[-1,1] has mean0 and probability1/2 at cut0; reflection swaps probability to1-P and negates the mean; cut outside interval yields0/1; empty intervals and zero direction length are rejected. No model probes or parameter tuning.

Competing explanations: (1) prior positive plug-in biases are an inherent sign theorem; an opposite-sign exact conditional example distinguishes this. (2) opposite sign comes from crossing a gate or rank boundary; T138 exact convex-cell certificate excludes that. (3) the Gaussian law is incorrectly induced along the line; independent completion-of-square and direct Gaussian integration check it. This is a regular Gaussian conditional distribution on a fixed transverse slice, not E27's particular distribution over omitted coordinates. It cannot overturn E27's60 empirical positive outcomes.

## Status
Protocol frozen; implementation next.

## Results
Pending.

## Compute accounting
Zero new neural forward passes or random inputs. Exact rational dot products and two high-precision scalar-CDF evaluations per frozen interval, plus controls. Record failed stages and CPU. One thread,2GB, review lock <=20CPU. This is a conditional component, not full estimator cost.

## Claims
[inferred] The conditional line law is Gaussian before truncation. [speculative] The two frozen intervals may produce opposite mean-replacement bias signs.

## Alternatives not ruled out
Integrating discarded-coordinate laws, weighted continuous symbolic states, alternative transverse parameterizations, and conditional distribution approximations. No global challenge accuracy or cheap integration of transverse coordinates is supplied here.

## Failure modes
Confusing a regular conditional slice with a positive-probability full-input event, losing Gaussian normalization/Jacobian, tail cancellation, treating precision agreement as an enclosure, or using local bias signs to refute a particular global mean-field experiment. Continuous perturbations of transverse coordinates preserve strict cell/output structure, but no global mass is computed.
