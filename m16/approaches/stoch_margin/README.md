# stoch_margin (T1): stochastic reduction of the tie margin

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md)

## Question
Every deterministic summary of the low inputs overestimates accuracy (H-E10). Does it help to carry the low inputs
as a *distribution* instead? Condition on a skeleton S (positions and values of the top 2-3 inputs). The other
inputs are then iid N(0,1) truncated below the 3rd value. Compute the law of the residual tie margin
deterministically (convolution, cumulants or a small gate-pattern mixture), take q(S) = P(correct | S), and
integrate q over the low-dimensional skeleton space with deterministic quadrature. Can this beat sampling on
leaderboard v3?

## Pre-registered alternatives (written before the full diagnostic; AGENTS.md rule 3)
The reduction needs q(S) correct to about 0.1 pp in aggregate. Possible reasons it could fail, with the cheapest
test that separates them:
1. **Non-additivity (gates).** The low inputs interact through ReLU gates, so the margin given S is not a sum of
   independent per-input terms. Test: a paired cut-HDMR additive surrogate (each 1-D term exact) against the model
   on the same low draws.
2. **Non-Gaussian shape, not interactions.** The margin given S may have the right first two moments but a shape
   (skew, atoms, multimodality from gate switches) that a 2-moment closure misses. Test: a Gaussian with the TRUE
   conditional mean and covariance of the 9 margins (a ceiling for any 2-moment method).
3. **Skeleton too coarse.** The error may be driven by one or two low inputs (the 4th largest sits near v3), so no
   noise model of the low inputs is accurate unless they enter the skeleton. Test: repeat 1-2 with a top-4 and a
   top-5 skeleton.
4. **Cost, not fidelity.** Even an accurate q(S) may cost too much: 720 position triples times a 3-D value
   integral, at roughly 10-20 forward-pass equivalents (FPE) per node, fits poorly into 16k-64k FPE. Test: count
   the nodes that a smooth-q quadrature needs per triple.

## Method
(in progress)

## Status
WIP. Diagnostics 1 (fidelity at K=3), 1b (K=4,5) done; diagnostics 2 (gate patterns) and 3 (anchor mixtures) running.

## Results
(in progress)

## Compute accounting
Diagnostics use random inputs for diagnosis only and are not charged to any estimate.

## Claims
(in progress)

## Alternatives not ruled out
(in progress)

## Failure modes
(in progress)
