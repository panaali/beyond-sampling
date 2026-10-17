# Can correcting Gaussian event overlap rescue the frozen candidate?

## Question
Can exact multivariate-Gaussian orthant integration fix E30 without changing its
current marginal failure probabilities?

## Method
For competitor failure events with Gaussian marginal probabilities q_j, any joint
law obeys max(q_j) <= P(union) <= min(1,sum(q_j)). Averaging over chambers gives
an interval of possible accuracy [est, est + union_gap] already stored by E30.
Choose the closest accuracy to the cached truth anywhere in that interval,
separately for each model, with zero extra compute. This oracle is more generous
than any implementable dependence correction. Compare its truth-variance-corrected
squared error to the mean per-model sampling MSE at the candidate's actual cost.

Alternatives before this check: (1) summed failure tails double-count overlap;
(2) the Gaussian marginals themselves are wrong because they ignore rank/gate
constraints. The interval separates whether overlap correction alone could suffice.
A further stress check moves every cached truth favorably by five standard errors.
This is a sensitivity calculation, not a simultaneous confidence bound.

## Status
Done. Overlap correction alone cannot rescue this frozen candidate on the
original or seed groups at any tested level. It does not distinguish frozen-gate
error from incorrect marginal distribution shape.

## Results

`python m16/approaches/crossover_compute/union_ceiling/check.py`

At actual costs 587.544 / 1174.944 / 2349.744 FPE, the original-model oracle MSE
ratios are 6.52 / 8.94 / 46.37 and seed-group ratios 35.24 / 84.91 / 172.81.
All six remain above one after a favorable five-standard-error truth shift.
The original truth exceeds the upper accuracy bound by 2.23 / 1.84 / 2.97 pp.
`results.json` retains all groups, including groups where this oracle could win.
The parent independently reproduced the reviewer's raw-only calculation.

## Compute accounting
Zero new model calls, input generation or LP solves. Arithmetic over 90 existing
rows; under one second of host work. Oracle integration is optimistically free;
actual integration could only increase cost.

## Claims
- [measured] The frozen candidate's overlap-correction ceiling still loses on
  original and seed groups at each tested level.
- [inferred] At least the marginal distributions or their frozen-gate maps must
  change; exact overlap integration by itself cannot solve this candidate.
- [speculative] Rank/gate-respecting non-Gaussian marginals may do better.

## Alternatives not ruled out
Different marginals, gates, chambers, refinement ladders, or a different estimator.
The calculation does not test a broader impossibility result or unseen model sets.

## Failure modes
The interval is valid only for the recorded approximate marginal probabilities;
it does not bound the true neural network. Cached truth uncertainty is imperfectly
known, and five-SE shifts are a sensitivity test rather than a formal confidence
statement. Oracle selection uses the truth, so it is not a deployable estimator.
