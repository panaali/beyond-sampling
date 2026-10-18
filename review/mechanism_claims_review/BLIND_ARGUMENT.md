# T201: what the blind-region argument proves

Assume ten independent continuous centered Gaussian inputs and let K count
strictly positive values. Signs are independent fair bits, so
P(K=0)=1/1024, P(K=1)=10/1024 and P(K<=1)=11/1024.
The second largest is nonpositive exactly on K<=1, except null zero/tie sets.
This is an input-distribution fact, independent of every network weight.

The conditional error is a separate, model-dependent quantity. Write e0 and
e1 for the error probabilities conditional on K=0 and K=1. Then the blind
contribution to total error is (e0+10 e1)/1024 and conditional blind error is
(e0+10 e1)/11. These equalities require no independence of the model prediction.

Under an ADDITIONAL genuine rank-blindness assumption, K=0 has ten exchangeable
possible target labels. A prediction independent of their negative ranks has
accuracy1/10, error9/10. For K=1, conditional on the positive location p, the
target is uniformly one of the other nine locations. A prediction that excludes
p and is independent of the negative ordering has accuracy1/9, error8/9.
If it sometimes chooses p, accuracy is instead (1-P(prediction=p))/9 under
that same independence premise. Thus even this favorable blind idealization
has combined error (9/10+10*(8/9))/11=881/990, not exactly8/9.

Neither no biases nor ReLU implies the required independence. A negative input
weight retains negative input magnitudes: ReLU(-x)=-x when x<0. As a universal
architecture claim, error8/9 is already refuted by all-zero weights with
first-index argmax: the constant prediction has conditional error9/10 on both
sign-count sectors by permutation symmetry. This is a theoretical counterexample
to the purported implication, NOT a result about the trained example model.
The actual trained error requires its own evidence or weight-specific theorem.

FINDINGS reports rounded e0=.889,e1=.768, without recovered sample counts or
per-input outputs. Substitution of these displayed numbers gives conditional
error .779 and unconditional blind contribution .0083681640625, consistent
with the displayed roughly.83% bucket. By comparison11/1024*8/9=11/1152
is roughly.95486% of all inputs. These are arithmetic contractions of rounded
claims, NOT independently measured rates, confidence intervals or a significance
test. Missing uncertainty and the model identity must stay explicit.

## Existing probe source is not an exact conditional average

Frozen `m16/estimate.py` hardcodes8/9 in several estimator variants. The later
`m16/arc_eval.py::_blind_probe` substitutes ten designed K=1 inputs. It takes
mean order statistics from the fixed model-free `_ref` sample conditioned on
exactly one positive, places the largest at each position p, and assigns the
remaining ranks in increasing label order. The variable `neg` is computed but
unused. No K=0 candidate is evaluated; only one negative ordering per p is
tested. Therefore these ten outputs do not implement (e0+10 e1)/11 or an exact
average over the9! negative orderings and continuous magnitudes. They remain
designed probes, with no universal conditional-accuracy guarantee. Importing
this module would generate400,000 reference inputs and load the model, so this
review reads its source without importing it.

## Proposed DRI wording

“The sign-defined region has exact mass11/1024. Its conditional error is
model-dependent. The reported empirical rates were.889/.768 for zero/one
positive input; their original counts and evaluation provenance are not yet
recovered. The8/9 factor is a heuristic requiring additional rank-independence
assumptions, and does not follow from the bias-free ReLU architecture.”

No shared narrative is edited here; independent review and DRI adoption remain.
