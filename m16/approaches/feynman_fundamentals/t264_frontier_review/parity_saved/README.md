# Independent T264 saved storage-parity audit

## Question
Did the storage-only rewrite retain all previously audited fixed-control outputs?

## Method
Checker6b2d3fa0 compares canonical full JSON for12 returned rows (removing only the new equality-status field) and all6 guard records. It inherits exact arithmetic solely through the previously verified raw digest, then independently checks source/gate/protocol hashes,18/12 counts,38 ordered producer events and actual resource limits. Neither helper is called. Source review56 d2b15bc6/shared702ac12e preceded immutable gate1fc1c4ef/sharedab0a90ce and the sole saved run. Producer raw3f7209dd, execution gate339c4621 and assignmentd7a2aa0f remain preserved.

## Status
Complete/pass. No failed attempt in this bounded saved review. Retained target data remains unopened and requires its separate source/resource gate.

## Results
[measured] All12 raw outputs and6 rejection records match exactly, including every order, rational value, midpoint decision, prefix selection and empty/tied case.18 attempts,12 returns,38 producer events and20 reviewer events reconcile. Comparison includes canonical value types; no numerical tolerance or rounding waiver.

Exact command used once; existing outputs prevent replay:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'set -e; set -o noclobber; ulimit -Sv 2097152; ulimit -Hv 2097152; ulimit -St 20; ulimit -Ht 20; flock /workspace/work/codex-review.lock timeout 25 /usr/bin/python3 /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/parity_saved/check.py > /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/parity_saved/run1.log 2>&1'
```

## Compute accounting
[measured] Reviewer main CPU0.030096024s; new producer main CPU0.095201595s separately. Actual reviewer soft/hard CPU20/20s and address space2147483648/2147483648 bytes are saved. Hashing/JSON/comparison/serialization are review overhead; imports and final serialization are outside main timer but within OS caps. Zero helper, target payload, model, mass, geometry or RNG calls.

## Claims
[inferred] Fixed saved parity agrees with the separately reviewed general prefix-length storage invariant. No empirical speed, full operation-price, integer-optimality or target accuracy/mass claim.

## Alternatives not ruled out
Other branches or future target schemas may expose defects. Historical floating skip_at_1se output is distinct from the new exact, index-tiebroken comparator. Retained numerical errors are not automatically true masses.

## Failure modes
Reusing an unbound prior artifact, changing prefix order/ties, dropping failures or extrapolating finite control parity to a deployable estimator would invalidate the stated scope. No automatic numerical continuation is granted.
