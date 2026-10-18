# Independent T237 capacity-study source and inference review

Scope: documentation, source text, hashes and Git tree inventory only. No training/model imports, checkpoints, arrays, random inputs, GPU/Colab work or MC evaluation. This review does not alter Claude's authorized T234 plan or budget and grants no separate numerical scope. The accompanying JSON identifies the exact reviewed versions.

## Source and provenance

The actual manifest contains three source snapshots plus a separately hashed directive. Its initial request's description of seven source bindings was inaccurate; review follows the manifest schema. All three snapshots agree byte-for-byte with both the clean installed alg-zoo checkout and the cited Git blobs at4433f505. The frozen branch f02482d6 has no entries under m16/approaches/capacity_ceiling. This is a bounded historical tree observation, not a claim about unpublished work or a later branch. The installed editable source is not proof of the eventual Colab execution source.

The continuous distribution task selects DistRNN. With no biases, H16/T10 uses16 input,256 recurrent and160 readout entries, totaling432. Its ReLU RNN is called without an initial-state argument, then its final state is read out. The task's argsort(-1)[...,-2] and prediction argmax semantics are visible; source-level review does not replace executed library tie semantics or cross-platform determinism evidence.

The training defaults, AdamW choice, learning rate .04/sqrt(H), weight decay .001, batch1024, 2^30 training sequences, 2^24 validation sequences and schedule parameters agree with the notes. The scheduler uses n_steps=n_train//batch_size in its exponent. Holding the named defaults while increasing n_train changes the entire decay history at a given absolute step. An earlier checkpoint of a longer-horizon run is therefore not interchangeable with a standalone shorter-horizon run. This is a legitimate combined intervention, not an isolated effect of more data.

The existing saving helper writes model.state_dict only. It cannot certify optimizer, scheduler, RNG or stream-position continuity. A weights-only restart must be described as such; this review makes no claim that the unpublished T234 implementation uses it. Default validation is smaller than the owner's required final evaluation, and both training and validation consume the evolving RNG without a separately named evaluation generator. Disjoint draws are not evidence of stream provenance, although this is not proof of dependence in an ideal iid model. Actual floored batch counts, source/library/precision identity and failure records are needed for later attribution.

Hard-case oversampling changes the expected unweighted cross-entropy objective. Exact importance weighting can restore the original population objective when support and weights are appropriate, while optimization variance, regularization and compute may still differ. Gaussian test accuracy must be evaluated under the requested Gaussian law; an enriched training score cannot substitute for it.

## Statistical inference

Let C be the supremum accuracy over the declared architecture and execution semantics. For each feasible fixed model, its accuracy is at most C. Thus a valid lower confidence bound for that model also lower-bounds C with the same coverage. Its upper confidence bound does not upper-bound C. A finite search, finite seed set or plateau therefore establishes best-found results or properties of the specified training procedure, not an architecture ceiling. Exact-real versus floating-execution semantics must not be silently mixed in applying this implication.

Conditional on a checkpoint fixed before an independent iid test stream, its correctness indicators are Bernoulli, so the binomial and SE formulas apply. The worst-case SE at10^8 samples is0.005 percentage points. This is conditional MC uncertainty, distinct from training-seed variability. Independent-stream difference variance is the sum of the two MC variances conditional on both models. Common-input differences additionally involve their correctness covariance, recoverable from paired outcomes or the full discordance counts. Sharing a training seed does not imply sharing test inputs.

A stopping rule based on two small observed gains is operational. An upper one-sided bound below the material-gain threshold supports no improvement of that size; it does not imply equivalence or exclude deterioration. The repaired text explicitly reserves equivalence for a confidence region wholly inside a declared two-sided margin. Neither conclusion establishes a permanent optimization plateau or a representation ceiling.

Pointwise intervals after selection/optional stopping are not automatically simultaneous. Predeclared simultaneous coverage, conditional per-look error spending on fresh streams, or fresh final evaluation of a locked selected model are valid ways to state an appropriately bounded claim. The finite training-procedure estimand and the population across future seeds require their own uncertainty assumptions; five seeds are not turned into an architecture bound by precise MC.

For iid centered Gaussian signs atT10, K_positive<=1 has probability11/1024. That is an input-event probability, not an architecture error floor or its conditional error rate. Overlapping blind/tie anatomy categories cannot be summed as if disjoint. Fixed precedence or a complete intersection table, with numerator/denominator accounting and rare/empty-bin uncertainty, makes the intended decomposition assessable. Co-area face contributions are not the same strata as events of actual Gaussian inputs.

## Disposition

The corrected written contract is sound within these limits. It identifies evidence needed to interpret the owner-authorized study without blocking or rewriting it. No training outcome, runtime provenance, convergence, improvement, universal ceiling, ARC rung or full-compute claim is verified here. Later driver/results require a separately scoped saved-evidence review; default review should not rerun training or MC.
