# T237 source observations for the owner-requested T234 study

Scope: claim e823d1d3, source/text only. T234's training and Colab owner remains
Claude; this review does not modify the authorized plan, budget, stopping rule
or runtime. Recommendations below specify what evidence supports each claim.

## Provenance boundary

Freshly fetched `claude/t192-gpu-cost-law` points to f02482d6 and has no
published capacity_ceiling files at the frozen inspection time. This is not
an assertion that the author has no unpublished implementation. The reviewed
training code is the clean local alg-zoo checkout4433f505 (2026-01-19), with
training.py last changed2025-12-15. The installed editable-package path was
read as text, not imported. The future Colab driver/runtime is unverified.
Manifest/snapshot bind the exact sources and T234 task-row request.

## Verified defaults and limitations

The source selects DistRNN for continuous distribution tasks. For H16/T10 with
bias=False it has Wi16x1, Wh16x16 and Wo10x16 (432 fitted entries), ReLU recurrence
and final-state readout. Training calls the default zero initial state. The task
uses argsort[..., -2], and prediction uses argmax. Float input ties and output
ties should retain the executed library semantics rather than being silently
changed for comparisons.

Training defaults match the board's AdamW base: lr_base=.04/sqrt(H), wd=.001,
batch1024, n_train=2^30. Additional defaults are warmup2^14 steps, minimum LR
multiplier.01, validation n_val=2^24, checkpoints10 and periodic logs. The source
sets torch.manual_seed and cuDNN deterministic flags, which do not alone prove
bitwise repeatability across GPU/library/precision changes.

**Increasing n_train changes the schedule as well as duration.** Its LR factor
is min(step/warmup,1)*max(decay_min_mult^(2*step/n_steps),decay_min_mult), with
n_steps=n_train//batch_size. Comparing default calls at2^30 and2^34 is therefore
a joint duration-and-relative-horizon-schedule intervention. It is a legitimate
experiment, but should not be described as data alone with identical LR history.
A checkpoint at2^30 within a2^34 run has a different preceding schedule from a
standalone2^30 run. This review does not require changing the owner's plan;
record which estimand the chosen implementation actually tests.

**The existing checkpoint helper saves only model.state_dict().** It does not
save AdamW moments, scheduler state, RNG state or data-stream position. Resuming
from weights alone is a new optimization treatment, not exact continuation.
A proper resume checkpoint or separately named restart arm can make either
choice interpretable. Never infer optimizer continuity from a matching filename.

**Default validation is2^24, below the requested>=10^8 final evaluation.** An
additional final evaluation path must be retained in the executed study record.
The source's validation follows training with the same evolving RNG state; it is
not an independently named holdout generator. Disjoint PRNG calls are not a
recorded independent stream. A fresh explicit evaluation stream makes the
selection/confirmation boundary auditable. None has been run by this reviewer.

Training uses unweighted cross-entropy on generated standard-normal inputs.
Hard-case oversampling changes the optimized risk unless importance weights
restore the original law. Either is an allowed treatment; disclose which one.
Evaluation remains N(0,I) as the owner requested. Cross-entropy/regularized-loss
plateaus are not the same statement as an accuracy optimum.

The source floors n_train/batch and n_val/batch. Preserve actual counts rather
than requested counts if a new driver permits nonmultiples. Checkpoint names
encode sequences used, but do not identify all training-source/optimizer choices.

## Minimum reviewable result record (no new execution request)

For every arm/seed/checkpoint: exact model and optimizer-source hashes; initial
checkpoint or initialization identity; seed/stream identifiers and training-law
mixture; actual steps/sequences; LR/weight-decay/schedule history and resume
semantics; checkpoint hash and dimensions; dtype/AMP/TF32/cuDNN/library/GPU
settings; wall/GPU/Colab units including interruptions and failed attempts.
Retain the owner's30-unit report and60-unit total limit. Do not silently discard
failed seeds, select only successful restarts, or call compute-cap termination
convergence. Existing source flags are evidence about configuration, not proof
of historical runtime equivalence.

Final evaluations need checkpoint hash, target law, requested/actual sample count,
stream identity, integer correct/error counts, and explicitly defined anatomy
counts. State whether an evaluation affected arm selection/stopping. Keep all
prior models and measures, even if a later arm is better. This is evidence
provenance, not an instruction to send data elsewhere or alter shared code.
