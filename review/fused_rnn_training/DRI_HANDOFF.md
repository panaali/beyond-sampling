# T261 integration handoff

The unchanged fused kernel passed all eight frozen CPU-interpreter local-update controls (six configurations, including batch 1024). Maximum absolute discrepancy across saved metrics is 2.384185791015625e-7; maximum relative L2 discrepancy is 2.8502714512413043e-7. Accuracy, counters and inactive parameter/moment equality passed. GPU behavior and speed remain untested here.

Published implementation: `review/fused_rnn_training/fused_step.py`. Against the pinned `trainer.py` at f8bbb07b:

```python
from fused_step import trainer_class
FusedTrainer = trainer_class(trainer_module)
tr = FusedTrainer(...)  # original constructor arguments; graph is a keyword
```

Specialization is H=16, T=10, float32 parameters/data, B<=1024. The factory retains original sampling and argsort outside the single forward/BPTT/AdamW compute kernel. Keep the original trainer module object: do not replace its `Trainer` attribute before calling the factory. Internal step counters are per model to avoid a cross-CTA race; scalar checkpoint export verifies equality, and inherited scalar restore broadcasts to all slots. CUDA graph capture, restore and actual-device behavior require device validation.

The CPU interpreter uses an explicit test-only NumPy pow/sqrt shim for otherwise unsupported double libdevice calls. The real kernel still calls CUDA libdevice. Sequence controls check three successive local updates from current fused state, not independent long-trajectory drift. Only summary error metrics are saved, so independent saved review checks those summaries and provenance rather than raw tensors.

Chronology and compute: attempt1 failed metadata validation before reference/kernel calls. Attempt2 completed one Torch reference and attempted one kernel, failing at interpreter libdevice dispatch after partial forward/BPTT work. Both are preserved. Attempt3 completed eight references and eight kernels, 1.163310262 internal CPU seconds (external 2.54 user +0.28 system). Cumulative scientific calls are nine completed references, nine kernel attempts, eight completed kernels. No GPU calls occurred.

Offline SM80 compilation is a separate bounded discriminator. Its first attempt failed before code generation because Triton's default home cache was read-only; preserved under compile_attempt1_cache_failure. The reviewed recovery relocates only the cache to workspace. Compile outcome will be appended here when available. Neither compilation nor interpreter parity demonstrates GPU occupancy or speed.

For a device benchmark, preserve TF32/IEEE settings and include setup, RNG/label generation, graph capture/reset and synchronization consistently. The one-CTA-per-model layout may have substantial register/shared-memory pressure, particularly B1024. A two-kernel reduction/update design may be faster; this prototype makes no throughput claim.
