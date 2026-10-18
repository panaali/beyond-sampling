# T261 independent fused RNN training step

## Question
Can a single compute/update launch replace the launch-bound H16,T10 forward, BPTT and AdamW step while preserving the existing trainer's semantics?

## Method
fused_step.py supplies FusedWorkspace.step and trainer_class(base_module). The latter retains original sampling/labels/CUDA graph code and changes only computation plus race-free per-model counter representation. trainer_snapshot.py pins the current interface. validate.py is a fixed CPU-interpreter/autograd comparison with explicit unsupported-libdevice shim. PROTOCOL.md freezes scope, resources, tolerances and exclusions.

## Status
Implementation and validation source frozen; independent56 kernel/adapter and910 harness reviews pending. No control run or GPU claim yet. Urgent deadline approximately05:40UTC, owner T192 independently implements and tests on Colab.

## Results
No numerical result yet. [inferred] One model per CTA avoids inter-model reductions and mutable shared-step races. [speculative] Fusion may reduce launch overhead, but register pressure, occupancy and IEEE dot throughput may dominate.

## Compute accounting
One forward/BPTT/AdamW kernel call per supplied batch; input RNG and argsort are retained outside. Eight fixed CPU validation/reference calls proposed, no training campaign. Buffer allocation is outside step and must be counted separately in any benchmark. No GPU available locally; interpreter does not validate device performance.

## Claims
[inferred] Drop-in constructor factory targets the frozen trainer interface; checkpoint step remains scalar on export, internal steps vector is documented. [speculative] GPU speedup remains unmeasured.

## Alternatives not ruled out
A reduction/update second kernel, persistent loops, different CTA tiling or original CUDA graphs may be faster. No broad negative follows a compiler/interpreter failure.

## Failure modes
Wrong batch gradient scale, state updates at lr0/inactive steps, TF32/FMA mismatch, shared-step races, graph restore, insufficient register capacity and interpreter-only validation. Retain all failures and distinguish numerical semantics from runtime performance.
