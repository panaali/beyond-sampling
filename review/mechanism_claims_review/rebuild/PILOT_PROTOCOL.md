# Fixed checkpoint wiring pilot

Claim50697a4c and reviewer delegation0ab7d45c follow the synthetic control
handoff274e739f and independent saved reviewa944f1aa. This protocol authorizes
no execution until the independent source/resource gate and immutable
`pilot_gate.json` are published. Exactly one run, no automatic retries.

## Fixed inputs and model

Use the original cached example checkpoint, SHA256
498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.
Require exactly three finite float32 CPU tensors: input16x1, recurrent16x16,
readout10x16. Convert stored binary weights to binary64 for both paths.
The model has no biases and uses ten ReLU recurrent steps from zero state.

The declared eventual streams use PCG64 seeds2026101802011 (train),
2026101802012 (holdout1),2026101802013 (holdout2), NumPy2.5.4.
Generate only the first1024x10 standard-normal values of the training stream.
Save its exact before/after PCG64 states, all inputs and their byte hash.
No seed, feature, tolerance, checkpoint or distribution selection is permitted.
Strict input ties stop without exclusion or resampling. No surrogate fit.

## Two implementations and retained evidence

The candidate is the previously reviewed `core.forward_final` plus the fixed
linear readout. The reference imports only the pinned original AlgZoo
architecture source, constructs its DistRNN on the meta device, assigns the
original stored state dictionary with strict key checking and assign=True,
then converts to CPU binary64 and evaluation mode. CPU RNG states must remain
identical through construction, assignment and forward; unsupported meta or
assignment behavior is a preserved failure, not permission to change paths.

The single library forward uses explicit zero initial state. A hook captures
the final hidden tensor from that same call. No second reference recurrence.
Retain all1024 candidate/library hidden vectors, logits, first-index argmax
labels, top-two gaps, true labels, input/model bytes and CPU RNG states.
NPZ files are exclusive immutable archives; results/progress record each stage
and failed attempt. Partial successful calls must not be silently rerun.

For each of hidden states and logits, require maximum absolute difference
<=1e-10 times max(1,maximum absolute reference value). Independently require
identical first-index winners. Save every mismatch and both gap arrays before
asserting parity. A small residual does not excuse a label mismatch. This is
a finite implementation check, not an outward floating certificate.

## Alternatives and resource accounting

Wrong tensor orientation, dtype, zero-state or readout wiring are discriminated
by the independent library recurrence and exact state-dictionary binding.
Different floating accumulation is separated from label parity by retaining
residuals and gaps. An input-stream/reuse mistake is discriminated by stored
PCG64 states and exact input bytes. Real-data rank deficiency and surrogate
quality are not tested in this pilot.

Exactly1024 distinct input vectors,10240 input normal values, one1024-sequence
candidate forward and one1024-sequence library forward. No fitted maps, MC
truth estimate, noise/model selection or accuracy conclusion. Imports, loading,
conversion, readout, sorting, serialization and verification are research work
and timed; canonical5948-per-input benchmark is not a complete cost charge.
Single thread, primary lock,2GB RSS limit and2500000KB virtual-memory shell
cap,110CPU external limit and90CPU soft guard. Use Bash builtin timing to
retain process timing; do not use the absent external /usr/bin/time.

Future training must load the saved prefix candidate arrays exactly once and
resume the saved PCG64 state for the remaining64512 observations. It requires
a separately published scope and independent pilot saved audit. Neither this
pilot nor the planned rebuild validates the old98%/90–91% historical rates,
proves summary sufficiency, or passes an ARC accuracy-versus-cost rung.
