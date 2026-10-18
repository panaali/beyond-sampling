# T278 independent saved primitive controls

## Question
Do the fixed exact partitions, signed interval differences and finite Gaussian Taylor diagnostics follow from their saved inputs?

## Method
`check.py` independently enumerates all 80 activation-mask assignments across four fixed networks and two depths. It reconstructs exact open-cell constraints, affine logits, every saved probe and frozen-mask result, first-index error intervals and endpoint-jump signed differences. It checks four Gaussian rows with independent 80-digit erf/exp expressions, including wide-band negative second-order terms. The fixed comparison is 1e-14 absolute for these bounded scalar diagnostics; original 3e-15 Taylor assertions remain unchanged. No author module is imported.

## Status
Frozen before any result/progress decoding. Independent source/resource gate and immutable completed-input binding required before a single saved audit. Producer source, raw publication and source review were announced, but outputs have not been inspected.

## Results
Pending. Intended sole run: one BLAS thread, review flock, CPU 19/20 seconds, address space 2097152 KiB, `timeout 25 /workspace/clean-neurons/.venv/bin/python check.py`, stdout/stderr retained in run1.log.

## Compute accounting
80 mask assignments, 56 saved direct/masked probe reconstructions and four zero-anchor mask setups. Four MP scalar references; zero producer/target/face/RNG calls. Named operation counts/peak bits are source metadata, not independently reproduced. Six malformed cases are checked as exact saved reasons, not rerun.

## Claims
[inferred, pending saved audit] Exact finite algebra can be independently checked without repeating producer execution. Finite Gaussian diagnostics are not outward certificates.

## Alternatives not ruled out
Real H4T4 face geometry, floating network calculations, Gaussian outer integration, model coupling and useful accuracy/cost remain untested here.

## Failure modes
Any pin, exact partition, probe, scalar or resource mismatch is preserved. No automatic retry or changed tolerance. Atomic partial results survive failures; no overwrite of prior audit evidence.
