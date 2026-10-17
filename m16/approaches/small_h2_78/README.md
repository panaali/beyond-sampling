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

### Recovery cycle 1 result
`check_prefix.py` reproduced the broad base/noise bounds and checked the constructive prefix integrals. There are49 rank-partition cases, 126 independent Bernoulli-convolution integrals (maximum disagreement1.12e-16), and327,680 conditional validation inputs (maximum|z|2.57). Three trained models, base and relative noise0.0025/0.02 from seed1, retain output support{0,5,6,7} and the reset theorem. Their broad interval is exactly[1111/14336,31071/71680]. All3x65,536 model-input checks obeyed the reset predictions. Input/output scale checks1e-16 and1e16 retained the support and reset-ray winner. These are author checks; the root owns independent review.

Reproduce from repository root:

```
flock /workspace/work/codex-compute.lock bash -c 'ulimit -v 2500000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/small_h2_78/check_prefix.py'
```

Saved `prefix_checks.json` includes checkpoint SHA256 and full counts. CPU1.052s, no analytic model-input samples; independent author validation used327,680 conditional inputs and196,608 full network inputs. Rational envelope operations and comparisons are reported separately; sorting/bit complexity remains uncharged, so there is no efficiency claim.

### Cycle 2 protocol, frozen before implementation
Constructively eliminate the final suffix input: split its scalar range at ReLU roots and readout crossings, then integrate the conditional rank law in v=Phi(x). Between fixed suffix CDF thresholds and1/2 that law is a polynomial of degree at most m+2. A fixed Gauss-Legendre rule of order ceil((m+3)/2) therefore evaluates the polynomial integral exactly in exact arithmetic; this model-free polynomial rule is distinct from approximating an RNN input integral. Validate against separate adaptive1-D integration on fixed suffixes, including predicted discarded indices and ties. Integrate only suffix lengths0..3 initially and report exact unresolved sign mass. Outer quadrature error must remain separately identified.

Alternative1: the rank polynomial degree or breakpoint set is incomplete. Test explicit Bernoulli integration and deliberately mixed-sign suffixes. Alternative2: gate/logit intervals miss atoms or tiny decision regions. Test constant duplicate readouts, rescaling, and independent dense/adaptive scalar calculations before any full-model claim. Compute cap remains120CPU seconds per cycle, serial lock and memory limit unchanged.
