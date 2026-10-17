# Independent first-sign history check

The T76 first-sign mixture's large failed T4 case is independently reproduced.
This checks a specific deterministic conditional-mean approximation, not a
general obstruction to using history or signs.

`check_sign_history_review.py` loads the frozen checkpoint and recreates the
RMS .02 copy0. It imports neither the author's affine clipping nor rank-law
implementation. Conditional two-step means are derived directly, followed by
two actual ReLU steps at a product midpoint rule in Gaussian CDF coordinates.
The one special prefix and one ordinary prefix rank probabilities are explicit
polynomial primitives. Correct and shuffled means use identical rank laws.

| CDF grid | Correct sign | Shuffled means |
| --- | ---: | ---: |
| 1024 x 1025 | .30975236115624016 | .28033039345912947 |
| 2048 x 2049 | .3097347191031055 | .28030857990964575 |

The author's values are .309720246156462 and .280284513892134. The finest
independent differences, 1.45e-5 and 2.41e-5, are small beside the correct-sign
bias of about .04605 against reference .2636679118915033. Shuffling reduces
this case's bias but leaves about .01664; it is a falsifier, not a validated
causal or deployable rescue. There is no error enclosure for this midpoint
quadrature, and both grid results remain recorded.

Probability partitions deviate by at most 2.22e-16 and never become negative.
Constant-label accuracy converges from .24999992060465584 to
.24999998014148084. Total independent work was 20,983,808 tiny two-step
forwards, 1.981502393 CPU seconds after imports. No random model-input samples
or fitted parameters were used. Raw outputs and checkpoint hash are retained
in `sign_history_review_results.json`.
