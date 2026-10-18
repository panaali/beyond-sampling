# T201 source and claim provenance

Claimcf62f4ab freezes the initial source-only phase. `pins.json` binds thirteen
snapshots to that shared commit. No frozen module is imported.

## The216-state lookup

`state.py` stores head top positions p1,p2,p3, magnitudes m1,m2,m3, last input
x9, and case0/1/2 (last input above/between/below the head top two). The key is
(case*9+p1)*9+p2. There are3*9*8=216 feasible states; the243-slot coding has27
impossible diagonal combinations p1=p2. All216 have open input realizations.
For continuous iid inputs, each state in cases0 and1 has probability1/720;
each case2 state has probability1/90. The states are not equally weighted.

The true target is already determined exactly by this state: p1 in case0,
label9 in case1, p2 in case2. “The answer depends only on the state” is therefore
a correct order-statistic identity. It does not prove that the network's hidden
state, prediction or error is determined by that state or by (state,m1,m2,x9).
A trivial target decoder has100% accuracy without reconstructing the network.
Consequently surrogate accuracy98% versus network95.3% alone is not evidence
of network equivalence; held-out hidden reconstruction AND agreement with the
network must be reported separately from task accuracy.

The encoder is present, but no lookup fit/evaluation driver, coefficient table,
train/test seeds, per-state counts or original model hash was recovered in the
searched tracked source/history. A general homogeneous linear map from three
features to16 hidden values would contain216*3*16=10,368 coefficients before
compression or constraints; an intercept would add3,456 more. This is a
description-size calculation, not evidence of the unrecovered original fit.

## Separable code

`separable.py` contains only `design(x,form)` and `fit(x,h,form,mask)`. It creates
position-one-hot multiplied scalar columns and calls least squares on the
supplied mask. It neither chooses the form/mask nor defines a train/test split,
performs evaluation, saves coefficients or prints the90–91% result. The stated
form m2*U[p2]+(m1-m2)*V[p1] has18 coefficients per neuron,288 total for16 neurons.
The fitted helper can implement it, but its existence cannot establish the
historical score or whether that score was held out.

`features.py` provides generic input/positive-part/leave-one-out/prefix-max
features, not a lookup experiment. `frozen.py` provides gate freezing and rank
canonicalization, not a lookup fit. `exact.py` reconstructs a sample's linear
logit map from its actual gates, not a saved error cross-tab. `common.py` loads
the example checkpoint at import without historical byte pins. Existing
`results/algzoo_16_10.json` reports the five-constant label experiment (replacement
accuracy.54133), not the98% lookup or90–91% separable experiment.

## Error buckets

FINDINGS lists global blind.83%, case2 picks third2.17%, case2 picks max.52%,
case0/1 mostly picks third.96%, other~.2%. RESULTS lists a differently worded
summary. Neither prose supplies a complete mutually exclusive table or counts.
The marginal percentages cannot reconstruct the requested case-by-pick cross-tab;
blind inputs can intersect each x9-case, so a new table must either include a
separate blind axis or state an explicit precedence before summing categories.
The displayed FINDINGS categories sum to about4.68%, close to its4.69% total;
the shorter RESULTS list (.82+2.14+.52+.27) sums to3.75%, about.94 percentage
points short of that total. These rounded prose contractions are not a recovered
cross-tab, nor proof the underlying samples/partitions were identical. They
motivate retaining the case0/1 contribution rather than silently dropping it.
Also “head third” and “global third” can differ when x9 falls between the head's
second and third values; the rebuilt table must define which ranking it uses.

A future fixed holdout should retain joint integer counts indexed by K=0/1/>=2,
x9 case0/1/2, and predicted GLOBAL rank1..10; this conserves every sample and
shows whether the.96% claim uses global third or head-third. Keep true/predicted
labels and sample/model/RNG hashes. This is a proposal only; no new forwards or
data generation are authorized by the present phase.

## Search scope and historical recovery

Read-only searches: `rg` over tracked root/m16/results Python, Markdown, text
and logs for discrete_state, separable, lookup,216, blind and reported rates;
`git log --all -G 'discrete_state|from separable|216.state' -- '*.py' '*.ipynb'`;
history of state/separable/FINDINGS; relevant original commit stats. Encoder
origin6cc72d88 adds only state/frozen;0c2ec2e1 adds separable/exact/modes;
dd8b65b9 adds the narrative. Available all-ref matching Python/notebook history
found only the two source origins. A wider filename search under workspace
mostly found duplicate worktrees. This is NOT proof that no notebook, untracked
scratch script, remote session transcript or other archive ever held the driver.

Two competing explanations remain: a valid independent fit was run in an
uncommitted session, or training and evaluation reused samples. Cheapest
discriminator is the original driver/raw artifact; absent that, a new explicitly
frozen held-out rebuild can test the mechanism but cannot retroactively validate
historical provenance. Neither historical98% nor90–91% should be advertised as
independently reproduced yet.
