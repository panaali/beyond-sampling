# minimal_pair (T10): which error patches appear at H=16, T=5 that do not exist at T=4?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md) ·
[sci_tao](../sci_tao/README.md) · [ht_seeds](../ht_seeds/README.md)

## Question
The deterministic tie-face calculation (sci_tao / ht_seeds) beats sampling by 56-109x at H=16, T=4 but only by
4.7-15x at T=5 (5 seeds, sci_tao's own cost count; the leaderboard charges that code 2.23x more). Which error mass is
new at T=5, where does it sit in input space, which neurons/gates make it, why does the T=4 calculation mis-weight
it, and can a frozen deductive recipe fix T=5 on all 5 seeds?

## Pre-registration (written 2026-10-18, before any diagnostic or scored run)

### Models
`zoo_2nd_argmax(16, T, seed=s)`, T in {4, 5}, s = 0..4 (10 models). Diagnostics on these; tuning (if any constant
is needed) only on T=4 models and on the leaderboard DEV models (`python m16/leaderboard.py dev --H 16 --T 5`).

### Diagnostics (D1-D4), same for every model
- **D1 error taxonomy (sampling, 2M inputs per model).** For each misclassified input: the true rank of the
  predicted index (1, 3, 4, 5), positions of ranks 1-3, gaps g12 = x(1)-x(2), g23 = x(2)-x(3). Report error mass
  per (predicted rank) x (position pattern), and the share of error mass that is "far from a tie"
  (min relevant gap > 0.3).
- **D2 face decomposition (64k iid faces per kind per model, plus the existing Sobol rule).** Per face record the
  configuration (a, b, p1), tie value u, window tau, number of linear pieces, and F. Decompose E[F] and E[F^2]
  (the variance driver) by configuration, by tau, by u, and by the rank of the wrong output.
- **D3 T=4 vs T=5 matching.** A T=5 input whose first element is "irrelevant" (x_0 below the 3rd largest of the
  rest) is the natural analogue of a T=4 input. Compare per-configuration E[F], E[F^2] between T=4 and T=5
  restricted to configurations where the T=5 extra position holds a low value; the residual is the "new" mass.
- **D4 mechanism.** For the faces carrying the top 1% of E[F^2]: gate patterns at the tie point, which neurons
  flip along the segment, and the sign/size of the readout margin. Compare to faces with typical F.

### Hypotheses and their alternatives (rule 3)
- **H1 (memory of early inputs).** New mass sits in configurations where the tied pair (or the rank-1 element)
  comes early, so its value must be carried through more steps; the readout drifts and the error band widens.
  Prediction: E[F^2] concentrates on configurations with min(a,b) or p1 small; within those, F grows with tau.
- **H2 (third-party confusion).** New mass is errors where the output is neither tied element (predicted rank 1 or
  >= 4). The tie-face segment does not localise these, so F is large on wide-window faces regardless of position.
  Prediction: D1 shows a jump in predicted-rank-not-in-{2,3} share between T=4 and T=5.
- **H3 (roughness, not location).** The new mass is spread over all configurations, with more gate flips per
  segment; the variance comes from fine structure in F, not from a few identifiable patches.
  Prediction: E[F^2] is not concentrated by configuration or tau; the top-1% faces have more cells, not
  special configurations.
- **Cheapest separating test:** the D2 E[F^2] decomposition by configuration and by error rank (one run per model,
  ~1 CPU-minute). H1 predicts concentration by position, H2 by wrong-output rank, H3 neither.

### Fix candidates (tried in this order, each a frozen recipe; nothing tuned on the test seeds' outputs)
1. **Configuration-stratified two-stage allocation:** stratify faces by position configuration (and window
   bins), pilot stage, then Neyman allocation per stratum from the pilot; all pilot costs charged.
2. **Region-targeted slicing:** if H2 holds, add a second slicing family across the tie relevant to the wrong
   output (e.g. 1/2 tie for rank-1 confusions) so each error patch is sliced by a segment that crosses it.
3. **Weight-derived stratification** (D4 mechanism -> a weight-only predictor of large-F faces) if 1-2 fail.

### Scoring (pre-registered pass criterion)
- `python m16/leaderboard.py truths --H 16 --T 5` then `python m16/leaderboard.py run minimal_pair --H 16 --T 5`
  (budgets 16k, 64k FPE; all groups: best-of-5 base, seeds 0-4, noisy copies), cost counted as implemented.
- **Pass:** MSE ratio vs sampling < 1 on each of seeds 0-4 at both budgets (point estimate), and the seed-group
  cluster-bootstrap CI upper bound < 1 at 64k. The same recipe is also run on H=16, T=4 (ladder rule 3: earlier
  rungs must still pass). Anything else is reported as a fail, with the numbers.

## Method
(filled in as work proceeds)

## Status
Pre-registered; diagnostics not yet run.

## Results
(none yet)

## Compute accounting
Unit = one forward pass of one length-T sequence (FPE), leaderboard rule 1. Walk cells counted as in sci_tao
(value + tangent recomputed from the flip step); the leaderboard's 2.23 cost factor for sci_tao applies to any
reuse of that walk until the implemented cost is shown to match.

## Claims
(none yet)

## Alternatives not ruled out
See hypotheses above.

## Failure modes
(none yet)
