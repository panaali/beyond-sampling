# T261 complementary kernel and adapter source review

## Question
Does the proposed fused supplied-batch step implement the frozen H16,T10 recurrence, cross-entropy gradients and AdamW schedule, with a compatible trainer lifecycle?

## Method
Read frozen fused_step.py and the exact trainer_snapshot.py (owner f8bbb07b), inspect each pointer shape and algebraic update, compare the snapshot to its Git blob without importing either source. recover_910 separately reviews the control/autograd harness. No duplicate implementation or numerical execution by this reviewer.

## Status
Written kernel/adapter source review passes for the declared finite valid-input controls. Final source_gate.json binds the final manifest/protocol and is conditional on completed independent harness review and immutable published execution gate. This is permission for the frozen bounded validation, not source adoption or a training campaign.

## Results
For each model, logits are h_T Wo^T, and dlogits=(softmax-onehot)/B; there is no K divisor. The reverse recurrence dz=dh*(h_t>0), gWi+=sum_b dz*x_t, gWh+=dz^T*h_(t-1), dh=dz*Wh and gWo=dlogits^T*h_T has the correct orientations. Weights remain unchanged until all gradients are formed. Padded classes are excluded from softmax and gradients; padded batch rows are excluded from losses/accuracy and derivatives. The scratch barrier precedes all backward loads from the changed layout.

Each program owns its model parameters, moments and step slot. No program reads the scalar another program increments. Step0 has zero learning rate but active moments update. The schedule uses the old completed-step value; bias corrections use step+1. The active mask freezes finite parameters/moments after n_steps while counters continue. FP64 schedule/powers then FP32 casts follow the reference precision intent; exact CUDA libdevice or compiler operation parity is not proven here.

Adapter construction disables base graph capture until the vector counter/workspace exists. Warm/n_steps/lr0 are made contiguous, satisfying pointer indexing. Inherited warmup resets the entire vector. Inherited scalar checkpoint copy broadcasts into the allocated vector; state export checks equality before returning a scalar clone. These preserve the source-level reset/load/graph-storage contract, subject to actual adapter controls. Supplied x/labels and inherited sampling/argsort remain outside the one compute/update launch; total training is not claimed to be one GPU launch.

## Compute accounting
Text, exact-byte/hash and AST inspection only. No Torch/Triton/author imports, model/autograd/optimizer/RNG or interpreter calls by this reviewer. The implementation's runtime/library provenance and future GPU measurements remain separate.

## Claims
[inferred] Finite valid-input algebra and source lifecycle are consistent with the frozen reference. [unverified] Interpreter controls must establish finite numeric parity. [unverified] GPU compilation, CTA visibility, libdevice accuracy, resource use and speed require CUDA tests. NumPy libdevice interpreter shims do not validate CUDA libdevice.

## Alternatives not ruled out
A whole-batch CTA may spill or underoccupy hardware; IEEE float32 dots may cost more than saved launches. Smaller batch tiles plus a reduction/update pass or other persistent layouts remain alternatives if actual compile/parity/profile fails. No failed implementation is an impossibility proof.

## Failure modes
This gate assumes matching same-device parameter/moment/scratch buffers, valid int64 labels in0..9, finite supported optimizer state, positive schedule denominators and the declared hyperparameters. The raw workspace API does not exhaustively validate arbitrary aliases, devices, values or malformed schedules. Floating reductions, exp/log/sqrt, compiler contraction and optimizer rounding may differ; frozen tolerances and all failures must be retained. The existing run metadata must additionally bind fused sources for deployment/checkpoint provenance. No historical CUDA binding or long-run training equivalence follows from source inspection.

## Separate offline compilation gate

Frozen b3915fcb is cleared only for two offline SM80 compilations (B32/warps8, B1024/warps16), after successful fixed interpreter validation and immutable published gate. Direct source review bindings, scope hashes, pointer types/constexpr choices and attempt/failure durability were inspected without importing or executing author/compiler code. The launcher must impose the recorded 105/110 CPU, 2GiB, primary-lock and 120-second wall contract. DIAGNOSTICS=False targets the production specialization; no CUDA launch, speed or device-equivalence claim follows. compile_gate.json binds exact bytes separately from the first validation gate.
