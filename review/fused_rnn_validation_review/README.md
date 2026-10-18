# T261 validation harness source review

## Question
Does the frozen CPU-interpreter harness test the declared fused training-step behavior against an independent Torch autograd/update reference without overstating CUDA validation?

## Method
Read only validate.py, PROTOCOL.md, manifest.json and the launcher, with the exact f8bbb07b trainer source as semantics reference. Kernel/adapter implementation is separately reviewed by recover_56. No imports or numerical calls by this reviewer.

Checks: per-model mean-B CE and sum-over-K gradient; strict ReLU derivative at zero; original-state independent forward/autograd; decoupled weight decay, moments and bias correction; zero-learning-rate first update with active moments; inactive bitwise state and incremented counters; odd-batch padding/zero-hidden/production-batch fixtures; finite absolute and relative-L2 criteria; first-index accuracy; attempt accounting, durable failures and bounded launcher; test-only symbol-specific NumPy pow/sqrt interpreter shims.

## Status
Final source3159f67c passes the written/source/resource review in source_gate.json. Clears only the fixed eight-call CPU-interpreter suite after separate56 kernel review and immutable gate publication. No reviewer execution.

## Results
Reference algebra matches trainer semantics. Final source includes the requested mandatory pins/reviewer completion records, pre-call attempts, final eight-call coverage, explicit resource enforcement and summary-only retention wording. The protocol explicitly distinguishes base lr0 from scheduled zero learning rate. The three successive calls each reference the current fused input state: this is local-update parity, not an independently evolving Torch trajectory test.

## Compute accounting
Zero scientific imports, tensor/checkpoint/model/fit/RNG/gradient/reference calls. Only source/Git reads, hashing and documentation. The author CPU-interpreter validation is a separate bounded eight-call run, subject to both source gates; no extra reviewer execution is authorized.

## Claims
[inferred] Source equations support the stated local tests if the frozen controls pass. No measured numeric parity or performance claim is made here.

## Alternatives not ruled out
Actual CUDA compilation, device libdevice differences, GPU races, register pressure/occupancy, throughput, adapter graph-capture/resume behavior and long-trajectory agreement still need distinct validation. NumPy shim success cannot substitute for GPU parity. Summary metrics do not permit independent reconstruction of all raw tensor comparisons.

## Failure modes
No passing conclusion follows from source alone. Preserve any runtime or numerical failure, actual attempt counts and resource costs; no tolerance change or dropped fixture. Kernel source changes require rebinding both reviews before execution. External data/labels remain real work outside the fused compute/update kernel.
