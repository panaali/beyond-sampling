# T49: H2T7 and H2T8 reset decomposition

## Question
Can exact resets remove most of the history from a deductive accuracy estimate for a two-neuron RNN?

## Method
Recover the earlier exact rational readout envelope and two-negative-input reset certificate. For checkpoint H2T8 seed0, partition by the last consecutive-negative pair, analytically marginalize all discarded prefix magnitudes using T55's formulas, then integrate only the remaining suffix. First-index output ties, including a positive-mass zero state, are retained. H2T7 has no standard zoo checkpoint; acquisition/training is separately tracked in T53 and no toy checkpoint is substituted here.

Recovery cycle protocol, frozen before implementation: verify exact envelope and reset coefficients; test prefix integration at fixed positive/negative suffixes and predicted labels both before and after the reset. Use exact polynomial antiderivatives in Gaussian CDF coordinates for discarded-index targets. Compare with independent conditional Monte Carlo only as validation. Subsequently integrate suffix lengths0,1,2 with analytic or deterministic low-dimensional integration and explicitly retain all unresolved probability mass. Test base seed0 and fixed relative noises0.0025/0.02 with RNG1 if checkpoints and time permit. No full accuracy claim from a partial decomposition.

## Status
Running; original uncommitted implementation was lost in the runtime restart. The earlier broad interval [7.7497%,43.3468%] is a recorded provisional result to reproduce, not a validated claim. Root reserved the task before delegation. Claude remains DRI.

## Results
Pending recovery implementation. Standard H2T7 full-training seed0 endpoint returned404 before the outage; zoo API explicitly excludes length7. H2T8 seed0 is the fixed primary checkpoint, not a best-of-five selection.

## Compute accounting
Each experimental cycle <=120 CPU seconds, one thread, process RSS below2GB. Numerical runs acquire /workspace/work/codex-compute.lock; long runs use ulimit -v2500000. Validation chunks <=65536. Count analytic preprocessing, exact rational operations, CDF/polynomial calls, suffix model steps, and independent validation inputs separately. Rational operations are not silently equated to FLOPs. No equal-compute advantage is claimed until cost is audited.

## Claims
- [inferred] If Wi>0, the first recurrent row is nonpositive and Wh[1,1]<=0, two negative inputs reset every possible nonnegative hidden state to zero.
- [inferred] Discarded values still affect the target rank; resetting hidden state does not permit deleting them from the accuracy calculation.
- [speculative] Analytic prefix marginalization may make the short suffix contributions cheap enough to assemble an accurate estimator.

## Alternatives not ruled out
1. A compact final readout may hide substantial rank dependence. Cheapest falsifier: fixed suffix integration over all discarded magnitudes, including predictions into the discarded prefix.
2. A reset partition may double count paths or drop the no-reset remainder. Cheapest falsifier: exact sign-pattern mass accounting and constant-readout tests.
3. Most mass may require long suffixes despite frequent resets. Cheapest test: exact reset-time distribution before expensive integration.

## Failure modes
Small numerical changes can alter positive-mass zero/tie behavior. Exact algebra alone does not certify floating-point integration. The no-reset remainder is non-negligible and cannot be discarded. Deterministic quadrature remains a separately classified numerical method; the exact prefix marginalization itself does not sample model inputs. Author checks do not constitute independent confirmation.
