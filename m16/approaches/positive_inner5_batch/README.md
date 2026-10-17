# T11589 batch primitive parity

## Question
Can the scalar positive-inner arithmetic be batched without altering the fixed primitive results, preparing a bounded dispatch-cost pilot?

## Method
Claim2557fa91, author34 yields. Float64 Nx5 array API, nonempty matching shapes. Finite inputs are scaled and processed by constant/same-sign/cross branches; all dot and wedge reductions retain explicit scalar left-fold ordering. Zero divisors reproduce the scalar arithmetic-failure status. No collinear fallback or tolerance change. Original26 primitive inputs/results are reused, plus4 new invalid-array guards. Supported outputs require identical float bits; all statuses/branches and original exception text retained.

## Status
Source/control protocol frozen; independent source gate before one30-control execution. No outer mass or trained run.

## Results
Pending. Any finite mismatch remains a failed control, not silently waived to a tolerance. API shape handling differs explicitly from original list-only scalar API; it is not an API drop-in replacement.

## Compute accounting
One thread2GB20CPU review. Candidate primitive and all control overhead recorded. Vector masks, indexing/copies/ufunc and reduction dispatch remain unpriced; no source-model FLOP saving or matched runtime speedup asserted. Prior scalar ledgers remain immutable.

## Claims
[inferred] Operation trees and left-fold reductions match scalar real arithmetic. [speculative] Current binary64 runtime matches saved output bits and batching reduces Python dispatch; only finite controls can test this.

## Alternatives not ruled out
NumPy versus scalar sqrt/divide, signed-zero, exception semantics and branching may differ outside these controls. Full outer construction/determinant/accumulation requires its own frozen pilot and current binary pins. No generic or full FP certificate.

## Failure modes
Pairwise sums replacing ordered additions, eager unsupported division, zero-scale underflow, changing branch inequalities, treating negative/nonfinite inputs as valid, or claiming native performance from a source operation count.
