# T477: exact integration transfer and deadline compatibility screen

## Question

Can radial averaging or exact line/circle integration improve tonight's strong
cumulant baseline without replacing it with an unvalidated estimator? Is the
current submission compatible with the actual flopscope client API?

## Method

Read-only source and mathematical review, followed only by arithmetic on existing
public reference summaries. No model evaluations or submission. Sources: prior
INDEX H-E17/H-A3, FINDINGS section 7 and DRI_HANDBOOK lessons; participant V29
findings F13/F62 and e30 public negatives; the current submission file; installed
whestbench 0.16.2; flopscope client source at
`a9f4deee64413b090447e51d7d45ddf33a7e17ab`.

Competing explanations and cheap discriminators:

1. Radius fluctuations dominate error: calculate the exact radial variance
   reduction from existing first/second moments, before evaluating any network.
2. Angular integration is cheap enough to justify its variance reduction: count
   the first layer's unavoidable circle breakpoints and cost a naive traversal;
   this does not bound efficient incremental traversal or its variance reduction.

## Status

Deadline screen complete: do not replace the cumulant candidate with an untested
line/circle estimator. This is a scoped scheduling decision, not a mathematical
impossibility result. API source audit found one nonfatal dtype-metadata mismatch.

## Results

For a bias-free homogeneous scalar output, write X=R U with R independent of U.
Replacing R by its exact mean reduces variance by
`Var(R) E[f(U)^2] = Var(R)/d * E[f(X)^2]`. Here
`E[R]=sqrt(2)*Gamma((d+1)/2)/Gamma(d/2)` and `E[R^2]=d`.
At d=1024, `Var(R)/d` is approximately 1/2048. The data-based arithmetic is
recorded separately in `radial_screen.json`; it is not an estimator experiment.
Exact radial averaging is established prior art, already recorded in T472.
Reproduce with `/tmp/whest-env-t473/bin/python m16/approaches/whitebox_lines/radial_screen.py`.

A generic circle crosses each first-layer nonzero ReLU hyperplane twice:
1024 neurons yield 2048 distinct crossings almost surely. A naive complete
downstream recomputation per arc already takes thousands of forward equivalents,
before extra deep-layer crossings. Incremental updates could be cheaper; their
cost and conditional variance reduction have not been measured. Our old successful
walks do not establish a benefit on this 1024-wide,16-layer task. Public Phase2
CV/QMC probes obtain modest sampling improvements against a cumulant frontier
hundreds of times better than sampling. This discourages a deadline implementation
without an independently checked cost/variance pilot.

### Submission/API review

Reviewed `whitebox_submission/package/estimator.py` as present at 23:40 UTC.
The client uses slotted RemoteArray objects; no custom array-attribute writes were
found. `as_symmetric` is a registered client proxy, with an integration test for
`symmetry=(0,1)`. Views, indexing, out buffers, QR, norm CDF/PDF and einsum are
client operations. Pure-Python os/environment configuration is permitted plumbing.

Nonfatal mismatch: RemoteArray.dtype is a string, whereas fnp.float32 is a dtype
label whose equality does not accept strings. `w.dtype == fnp.float32` therefore
causes redundant casts remotely. `str(w.dtype) == 'float32'` is metadata-compatible
with both local and client paths. Root was notified; no submission files edited.
The audit does not establish remote residual-time or runtime eligibility.

The installed whestbench subprocess runner expands every weight into Python
float lists, JSON-serializes them, then retains decoded request lists while
constructing worker arrays. At16.8M weights, these copies can add hundreds of MB
per process beyond estimator buffers. Its observed RSS stop is not evidence that
the estimator alone exceeds the cap. Record which PID/phase hits the guard.
The installed subprocess worker uses local flopscope; it is not a test of the
actual remote-client transport.

## Compute accounting

Zero model forward passes, quadrature traversals or estimator runs. Only source
reads and arithmetic on four saved vectors. Time cap3CPU-min,one thread,2GB.
An unimplemented next pilot is estimated at2-6agent-hours and0.1-2CPU-hours for
primitive checks and a small-width gate-count ladder; full-width cost remains
unknown. At an illustrative $0.10/CPU-hour this is $0.01-$0.20 today; with a
speculative 5x compute-price improvement, $0.002-$0.04 in five years. More compute
alone does not resolve the missing efficient representation/variance evidence.

## Claims

- [inferred] The radial variance identity is exact under positive homogeneity
  and isotropic Gaussian input, apart from floating arithmetic in its evaluation.
- [inferred] A naive circle implementation is a poor deadline candidate.
- [measured] Source inspection finds the dtype comparison mismatch and JSON
  weight-materialization path; no remote execution was performed by this agent.
- [speculative] Incremental exact-circle integration could have a useful niche.

## Alternatives not ruled out

No line/circle estimator variants were run. Incremental gate updates, adaptive
planes, angular control variates, integration of a cumulant residual, and cheap
partial-network conditional integration remain open. Scope of the public F62
negative is its finite tested CV/QMC families, not every sampling method.

## Failure modes

Gate proliferation; traversal overhead; radius already concentrated; angular
variance remaining after exact integration; cancellation between approximation
errors; and remote-client call overhead. Source compatibility is weaker than a
successful wire-runtime check. Our old deterministic quadrature successes are
scoped results, not an accepted general solution to either challenge.

## Source exposure (2026-10-21)

- [Client source, pinned](https://github.com/AIcrowd/flopscope/tree/a9f4deee64413b090447e51d7d45ddf33a7e17ab/flopscope-client):
  `_remote_array.py`, `_dtypes.py`, `__init__.py`, `_getattr.py`, integration tests.
  dtype metadata, supported proxy operations and slotted array compatibility.
- [V29 findings](https://github.com/504aldo/whest-p2-cumulant-k3/blob/a038d17924cd2df82dbf3a0d42da38fee48aa5a5/docs/findings_log.md):
  F13/F62 scope retained from earlier review; no new universal sampling exclusion.
- [e30 write-up](https://github.com/siming767-creator/whestbench-lordprocciuto/blob/3d038e9f48a97671878ee7bbef02e193f128b61c/WRITEUP.md):
  prior cubature/adjoint/oracle negatives, already exposed in the earlier review.
- Installed whestbench0.16.2 `runner.py` and `subprocess_worker.py`:
  Python-list/JSON memory overhead, local-backend subprocess execution.
