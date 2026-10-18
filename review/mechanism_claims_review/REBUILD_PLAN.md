# Proposed bounded held-out rebuild; no execution authorization yet

This document plans a possible continuation after independent review of the
source/theorem phase. It is not an executable protocol, claim expansion or
permission to run model forwards. A fresh TASKS scope and frozen source/resource
gates must precede implementation/execution as applicable.

## Decision the measurement would resolve

The original fit driver/split remains unavailable. A fixed new reconstruction
can establish its own held-out performance while explicitly leaving historical
provenance unverified. It can simultaneously provide the missing mutually
exclusive error cross-tab from one unchanged example model. This is a mechanism
audit, not a deductive estimator, new training of the RNN, or ARC rung pass.

## Proposed fixed method and data

Reuse cached original example checkpoint at
/workspace/work/t39_sources/example_checkpoint.pth, SHA256
498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.
Installed alg_zoo/loading.py identifies example_2nd_argmax with
gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth. The new protocol
must pin loader/checkpoint bytes without treating them as historical fit pins.
No download or checkpoint choice is needed if this existing identity verifies.

Use one predeclared65536-input training stream and two independent65536-input
test streams, all iid N(0,1)^10, generator/version/three seeds fixed before run.
Chunk forward computation at1024; retain chunk RNG/input hashes and model bytes.
Do not tune features, model, seeds, sample sizes, regularization or tolerances
after seeing test scores. No resampling if a training state lacks full design
rank: preserve incomplete/unknown status and stop dependent scoring.

Lookup: in each of216 states fit a homogeneous least-squares map from exactly
(m1,m2,x9) to the16-dimensional final hidden state. No intercept, ridge, clipping
or extra ReLU. All216 fitted matrices are saved. The source's “linear map” lacks
a recovered intercept convention; this choice is a new declared reconstruction,
not an assertion about the historical fit.

Separable: use exactly18 columns m2*onehot(p2) and(m1-m2)*onehot(p1), fit on
training case2 only, score on test case2 only. Save the18×16 coefficient matrix.
Both methods use the fixed original readout on the raw fitted hidden vectors.
Target labels play no role in fitting. They are used only for scoring.

## Outputs, denominators and uncertainty

Report train and each holdout separately. Include per-neuron hidden MSE/R2
(test-mean denominator, zero-variance handled explicitly), medianR2, agreement
with the network, task accuracy, and original network accuracy. Case2-restricted
and whole-input denominators must never be confused. A direct state-to-true-label
decoder should verify100% on strict inputs; it is an identity control showing
why task accuracy alone is not network reconstruction.

Save every integer cell of K-category(0,1,>=2) × x9-case(0,1,2) × predicted
GLOBAL rank(1..10). Save correctness and true/predicted labels sufficiently to
check conservation. All rows belong to exactly one cell; blind mass is not added
again to overlapping x9 buckets. Report the case0/1 third-rank error contribution
and blind-first disjoint decomposition explicitly, rather than reverse-engineer
the old.96% prose number. Report e0/e1 sample counts and uncertainty.

Binomial intervals describe fixed-model/fixed-trained-surrogate evaluation
uncertainty; they do not include training-set variability or validate historical
rates. No fitted best-budget selection, significance claim from rounded old
numbers, or accuracy/MSE-vs-compute conclusion.

## Gates and resource budget

Before the real model: independently review analytic state/design/lstsq controls
covering216 codes, impossible diagonal codes, target mapping, known exact
coefficient recovery, rank-deficient/empty design handling and aggregation.
Then freeze a small direct-recurrence checkpoint pilot and independent outputs
before the full continuation. Preserve all failures and successful pilot reuse.

At most196608 distinct input forwards in the full proposed design, including
any pilot reused from that design. Whole hidden-state arrays are unnecessary;
final states and sufficient regression/score records suffice. The canonical
5948-FLOP forward benchmark would contribute1,169,424,384 scalar units alone,
excluding fitting, scoring, generation, serialization and controls; this is
planning arithmetic, not a complete measured cost or challenge budget.
One thread,2GB,primary flock, at most110CPU per process with a frozen total
reserve; stop on any cap/failure. A source-first pilot may show a smaller total
reserve is adequate. No repeated trial to achieve a desired score.
