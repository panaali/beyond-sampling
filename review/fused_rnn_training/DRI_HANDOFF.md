# T261 current integration handoff

Use **tiled_step.py**, the fixed serial32-sample-tile rescue. It passed all eight frozen CPU-interpreter local-update controls and both offline SM80 compiles before the05:40UTC deadline. No GPU is available here: actual-device correctness, launch and performance remain untested. The original `fused_step.py` B1024 specialization is explicitly blocked by excessive shared-memory demand and is preserved as evidence.

## Integration

```python
from tiled_step import trainer_class
FusedTrainer = trainer_class(trainer_module)
tr = FusedTrainer(...)  # original constructor arguments; graph is a keyword
```

This targets the pinned original trainer.py at f8bbb07b, H16,T10,float32 parameters/data,1<=B<=1024. Keep the original module object: do not replace its Trainer attribute before calling the factory. The inherited sampler and argsort remain outside the single forward/BPTT/AdamW compute kernel. One CTA owns each model, loops over32-sample tiles, accumulates all gradients before updating weights once, and increments its own step counter. Internal counters avoid the cross-CTA shared-scalar race; export verifies equality and emits the original scalar checkpoint, while inherited restore broadcasts it. CUDA graph capture/reset/load and device behavior still need validation.

## Fixed evidence

- Tiled interpreter: six configurations/eight reference+kernel calls, including initial zero LR, mixed inactive models, odd batches, zero hidden states, three successive local updates and B1024. All gradients, parameters, moments, logits and losses pass unchanged thresholds. Maximum absolute discrepancy2.384185791015625e-7; maximum relative L2 discrepancy3.932716176130984e-7. Accuracy/counters/inactive state checks pass. InternalCPU2.634479931; external3.91user+.20system/4.11wall.
- Tiled offline CUDA SM80 codegen: B32 andB1024, fixedBB32/warps8, explicit IEEE dot and FMAfalse. Compiler metadata shared memory11776 and12800 bytes respectively. Both PTX and cubin hashes retained, PTX files saved. InternalCPU2.251590628; external2.67user+.22system/2.89wall. Zero GPU launches. Device registers/occupancy/speed were not measured.
- Original whole-batch interpreter also passed eight controls, but original SM80 B1024 codegen requested332800 shared bytes (B32:11776). Therefore original B1024 is not launch-ready. The new tiled results are separate; reduction order changed and prior parity was not transferred.

The CPU interpreter uses an explicit test-only NumPy pow/sqrt shim because its CUDA libdevice dispatch is unsupported. The actual CUDA kernel still calls libdevice. Sequence controls compare each local update from current fused state, not accumulated drift of independently evolving trajectories. Saved metrics support summary-level independent audits, not independent raw-tensor reconstruction.

## Preserved failures and cumulative accounting

Attempt1 failed metadata validation before scientific calls due to a transient reviewer-note mutation; original bytes were restored and the reviewer acknowledged it. Attempt2 completed one Torch reference and attempted one kernel, failing at interpreter libdevice dispatch after partial work. Attempt3 completed eight original-kernel references/updates. Tiled attempt1 completed eight more. Cumulative scientific totals:17 completed references,17 kernel attempts,16 completed kernels. Prior failures and raw logs remain immutable.

Original offline compile attempt1 entered compilation but failed before code generation on the read-only default home cache. Workspace-cache recovery completed both original specializations, exposing the shared-memory blocker. Two tiled specializations then compiled. Across compile phases: five entered compile calls, four completed code generations, zero GPU launches. Internal timers exclude imports/child CPU; external process logs remain separate. Do not convert these CPU timings into GPU estimates.

## Review and next device gate

Original and tiled kernel/adapter source reviewed by recover_56; validation/reference source and attempt provenance by recover_910. Frozen source, review and execution gates precede each run. Tiled source gates69190e6c and37a366ae; validation rawfc4d4707/shared744a9e6a. Saved independent reviews will be attached when published.

For the Colab device control, compare the same supplied x/y and initial weights/moments/counters against the original trainer before any timed training. Bind runtime/device/precision, preserve per-model schedule, test inactive models and graph checkpoint/reset semantics, then benchmark equivalent synchronized work including or separately reporting sampler/labels/setup/capture. This prototype makes no speedup or trained-accuracy claim. One CTA per model and serial batch tiles may limit occupancy/throughput; a two-kernel reduction/update implementation remains an alternative.
