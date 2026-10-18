# T235 independent saved synthetic audit

## Question
Do the frozen synthetic records satisfy the exact grid-law contracts, including flat-CDF endpoints and zero-probability rejection?

## Method
Delegated by `bad307f3`. The checker was frozen at `6723ca7c`, independently source-reviewed by h4 at `b21b4c1b` (shared `dacd19fa`), then its immutable execution gate was published at `9b060cae` (shared `1a86d6f7`) before the sole run. Author results `6de50d53` were already published at `94fa9f54`.

`check.py` integrates each interval's rational density to reconstruct CDF values, instead of the author's bisect implementation. It reconstructs the inverse through exact knot lookup or barycentric interpolation, verifies all bin masses and conditional coordinates, and independently checks the logical reason for every saved exception. It imports no author code. Source/protocol/reviewer pins, fixed coverage, saved fixtures and the complete progress sequence are checked.

## Status
Passed. One saved-only review attempt; no failures or retries. Author results unchanged.

## Results
All eight grids, sixteen guards, forty inverse probes, forty-six CDF probes, seventy-eight conditional records and fifty progress events pass. All arithmetic comparisons are exact rational equalities. The author's recorded CPU value is retained as metadata, not independently timed.

Exact command used once (output files intentionally refuse overwrite):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'set -o noclobber; ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock timeout 25 /workspace/clean-neurons/.venv/bin/python review/t235_primitive_saved/check.py > review/t235_primitive_saved/run1.log 2>&1'
```

## Compute accounting
Review CPU after standard-library imports: 0.003994168 seconds; peak RSS 25,180 KiB. External limits: 20 CPU seconds, 2 GiB address space, one thread, shared review lock, 25 wall seconds after lock acquisition. Zero author primitive, real face-grid, model, RNG, integration or mass calls. Metadata/hash verification and exact arithmetic are review work, not a production estimator cost estimate.

## Claims
- [measured] Every frozen saved synthetic arithmetic result and coverage/provenance contract checked by this reviewer passes.
- [inferred] The finite fixtures support the stated exact-rational implementation, including its explicit rightmost-knot endpoint policy.
- [speculative] A separately specified face-law implementation may use this primitive; no performance or accuracy result follows here.

## Alternatives not ruled out
Actual floating CDF arrays, rounding near endpoints, finite Sobol atoms and expensive setup may change usefulness. The real Gaussian law and the frozen-array real law remain distinct. No target-grid construction or data extension was evaluated.

## Failure modes
The fixtures do not establish universal software correctness, finite NumPy parity, numerical probability accuracy, or a cheap conditional estimator. Guard verification checks saved exception classes against independent logical invalidity; it does not rerun exceptions or independently prove timing/resource metadata. Known-law sampling remains a baseline rather than a deductive ARC solution.
