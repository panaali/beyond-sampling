# T141 conditional Gaussian retained-margin coordinate

## Question
Can retaining the exact scalar Gaussian law resolve the T138 same-gate ambiguity, and does mean replacement have a universal bias sign? Claim ec89608 was pushed before implementation.

## Method
Pin T138 exact pair and margin certificate. Set x(lambda)=a+lambda*d, where d=b-a and a is correct. Conditional on the Gaussian input's component orthogonal to d equalling this line's fixed transverse component, lambda has normal mean mu=-(a dot d)/(d dot d) and variance1/(d dot d). Condition further on one of the two predeclared intervals[0,1] and[1/2,1]. T138 independently proves all gates and ranks unchanged, with correctness exactly lambda<c for the rational crossing c, except the measure-zero output tie.

Use the analytic normal-CDF ratio to integrate correctness and the analytic truncated-normal mean for the mean representative. Freeze60/90-digit evaluations to check numerical stability; agreement is not a certified transcendental interval. Independent reviewer uses direct scalar quadrature. Analytic controls: standard symmetric interval[-1,1] has mean0 and probability1/2 at cut0; reflection swaps probability to1-P and negates the mean; cut outside interval yields0/1; empty intervals and zero direction length are rejected. No model probes or parameter tuning.

Competing explanations: (1) prior positive plug-in biases are an inherent sign theorem; an opposite-sign exact conditional example distinguishes this. (2) opposite sign comes from crossing a gate or rank boundary; T138 exact convex-cell certificate excludes that. (3) the Gaussian law is incorrectly induced along the line; independent completion-of-square and direct Gaussian integration check it. This is a regular Gaussian conditional distribution on a fixed transverse slice, not E27's particular distribution over omitted coordinates. It cannot overturn E27's60 empirical positive outcomes.

## Status
Done as a bounded component. T55 independently integrated the original unnormalized Gaussian density and its first moment at70digits, reproducing probabilities/means within9.1e-72, and independently checked every rational sign inequality. DRI registration requested.

## Results
The exact conditional law has mu=.41090321648160383 and variance=.607916929974621. On[0,1], integrated correctness is .724813275187512; its mean .488446562402731 predicts correct, so replacement bias is +.275186724812488. On[1/2,1], correctness is .429435506464070; its mean .738552114127653 predicts wrong, so replacement bias is -.429435506464070. These are conditional probabilities on two different truncated line slices, not full-model accuracies or contributions to E27's experiment.

The60/90-digit results agree within4.71e-56. Separately, exact rational exponent spreads .28542964/.27890059 imply density ratios less than2; normalized first-moment bounds are strictly negative(-.30709277) and positive(.01156351). Thus both bias signs follow without numerical transcendental assumptions. Quantitative probability values remain precision-checked floating results.

Analytic symmetry, reflection, outside-threshold and degenerate-interval controls all pass, as do three exact completion-of-square checks and exact orthogonality. Reproduce with single-thread `/workspace/clean-neurons/.venv/bin/python m16/approaches/gaussian_margin_slice/conditional.py`, then `sign_certificate.py` in that folder, using the project review lock and2GB limit. No failure was observed in this bounded phase.

## Compute accounting
Zero new neural forward passes or random inputs. Exact rational dot products and two high-precision scalar-CDF evaluations per frozen interval, plus controls. Record failed stages and CPU. One thread,2GB, review lock <=20CPU. This is a conditional component, not full estimator cost.

## Claims
[inferred, exact algebra] The conditional line law is Gaussian before truncation. [inferred, exact sign certificate independently checked] Mean replacement has opposite bias signs on the two frozen slices. [measured] High-precision CDF values are stable at the stated precision, without formal transcendental enclosures.

## Alternatives not ruled out
Integrating discarded-coordinate laws, weighted continuous symbolic states, alternative transverse parameterizations, and conditional distribution approximations. No global challenge accuracy or cheap integration of transverse coordinates is supplied here.

## Failure modes
Confusing a regular conditional slice with a positive-probability full-input event, losing Gaussian normalization/Jacobian, tail cancellation, treating precision agreement as an enclosure, or using local bias signs to refute a particular global mean-field experiment. Continuous perturbations of transverse coordinates preserve strict cell/output structure, but no global mass is computed.

### Frozen exact bias-sign check
The CDF evaluations suggest opposite signs. Before implementing an additional certificate, freeze a rational-only check: the Gaussian exponent range on each interval is less than1/2, hence density max/min is less than2 by exp(t)<=1/(1-t). For crossing c, c^2>2(1-c)^2 bounds the first interval's integral of(lambda-c) times density strictly below zero. On[1/2,1], (1-c)^2>2(c-1/2)^2 bounds it strictly above zero. These rational inequalities certify the mean's side independently of transcendental evaluation. Since c is interior, correctness probability lies strictly between0 and1; both bias signs then follow rigorously. Numerical probability magnitudes remain precision-checked approximations.

### Scope and alternatives checklist
Tested both predeclared slices, two precisions, elementary exact sign bounds and analytic controls. Untested: global weighting of transverse slices, actual E27 averaging distribution, other models/noise, estimator setup/integration cost and transverse quadrature. None are ruled out. This demonstrates a constructive way to integrate one residual coordinate and refutes only a universal sign theorem for arbitrary mean replacement. It neither explains nor contradicts E27's60 positive global biases. The strict gate/rank/output and sign inequalities persist on sufficiently small transverse neighborhoods by continuity, but their Gaussian mass is not computed.
