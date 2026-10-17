# T42: independent audit of the T35 cumulant wrapper

## Question

Does the DRI's RNN-to-feedforward adaptation preserve the scoring problem, and
do its conditional inputs and compute declarations match their stated meaning?
This supports Claude's T35 implementation; it is not another estimator.

## Method

Read-only audit at repository revision
`90608af64abd6da794af0b64f855572e3f4f4e99`. T35's last source commit at this
snapshot is `9ddfba282809d18def39a6e8a41caf90ef1af590`. Wrapper SHA256:
`59b1bab254f9da0ea63c8384567db5533985218e99e039352d00f43a7e305177`.

`check.py` extracts the actual pure wrapper functions by Python AST and executes
them unchanged. This avoids downloading missing optional dependencies
(`jaxtyping`, `einops`, `opt_einsum`, `joblib`). The vendor's own `get_d_max` and
primitive counter definitions are likewise extracted unchanged. There is no
full cumulant propagation, dependency stub, or change to T35/vendor/shared code.

Before tests, distinguish two alternatives:

1. A tie bug could be real but irrelevant to the observed DEV failure. Test
   identical-output synthetic models and count exact ties/mismatches separately
   on the two cached DEV models.
2. An apparently defective conditional-input construction could instead be a
   deliberate approximation. Separate exact index mapping and correct cumulant
   algebra from incomplete symmetry, finite-sample noise, and mode truncation.

## Status

Complete bounded review. Findings apply to the pinned snapshot, not subsequent
DRI edits. Claude remains responsible for acceptance and changes to T35.

## Results and actionable findings

### 1. Fix first-index logit ties before claiming exact readout equivalence

In `arc_kprop.py:94-102`, unconditional mode defines
`U_ij = 1[logit_i > logit_j]`, for i<j, then awards j the complementary win
`1-U_ij`. This breaks ties in favor of the larger index. NumPy argmax chooses the
smaller index. Cell mode at lines118-127 requires strictly positive margins
against every competitor, excluding valid first-index tied winners altogether.

Minimal model: H=1, T=3, Wi=[1], Wh=[[1]], Wo=[[1],[1],[1]]. Test the six
permutations of [1,2,3]. All logits are equal, so the correct model always picks
index 0. The unconditional stack instead picks index 2: **4/6 correctness bits
differ**, although both aggregate accuracies are 1/3. The cell stack gives
**accuracy 0 instead of 1/3**. Aggregate-only validation can hide the first bug.

A random H3,T4 check has 111/512 tied maxima, 69 unconditional mismatches and 32
cell mismatches. Every unconditional mismatch is on a tied maximum.

**Practical scope:** on 16,384 Gaussian inputs for each frozen DEV copy
(noise seed 1, sigma .0025 and .02), there are **zero exact maximum ties and zero
readout mismatches**. This generic correctness bug is not demonstrated to cause
the observed M16,T10 DEV propagation failure. The RNN hidden recurrence and
input pass-through agree with independent direct recurrence on every check.

Suggested correction: for i<j use `1[logit_j > logit_i]` as j's win and its
complement as i's win. In cell mode, require a strict win against smaller
indices and a non-strict win against larger indices, implementable as
`1-1[logit_i > logit_p2]`. Do not use a fixed numerical epsilon to change ties.
The corresponding cumulant endpoint also needs an explicit atom convention.

### 2. AUGMENT K2 silently discards available input cumulants

`cell_input` at line 242 hardcodes `KH.SIMPLE`, although `estimate_cell` accepts
`kind="AUGMENT"`. For K2, SIMPLE retains degrees 1/2 while AUGMENT requests
degrees 1..4. Even when a supplied base dictionary contains all four, the wrapper
passes only 1/2 into propagation. The tiny reproduction records this mismatch.

Pass the requested kind through `cell_input` and test retained degrees. This
finding specifically affects **AUGMENT K2**; SIMPLE K2 is consistent, and
SIMPLE/AUGMENT K3 both request degree 4. AUGMENT K4 requests degree 6, while the
current base construction is capped at 4; any such use must state the additional
input truncation or supply the missing degrees.

### 3. Conditional-precomputation edge cases and incomplete symmetry

The sampler draws the correct canonical conditional law: sorting puts the
largest two at positions 0/1, and an independent permutation restores
exchangeability of the lower coordinates. The raw-moment-to-cumulant formulas
through degree 4 and the inverse coordinate permutation in `cell_input` are
algebraically correct.

However, `cell_base_cumulants` loops over `n//200000` full chunks and divides by
the requested n. For nonmultiples, every raw moment is scaled by
`200000*floor(n/200000)/n` before cumulant conversion. The actual n=1 call
raises a scalar/einsum ValueError after drawing nothing. Process the last
partial chunk, or reject unsupported n explicitly. **The default 4,000,000 is
an exact multiple and is unaffected.** Invalidate caches made with affected n.

The symmetry code averages cyclic shifts and reflections, namely a dihedral
subgroup, rather than all lower-coordinate permutations. For four or more lower
coordinates this does not enforce full exchangeability. An explicit symmetric
T6 symmetric tensor retains a 0.25 discrepancy after swapping two lower
coordinates. For example, adjacent and opposite pairs on the artificial cycle
can remain different. This does **not** by itself bias the underlying raw
moment estimator: every group element preserves the true law. It leaves
avoidable finite-sample anisotropy; empirical cumulants also have their usual
finite-sample plug-in error. No effect on actual 4M tables was measured.

An efficient full projection can average entries by their equality pattern
among lower-coordinate indices, retaining the distinguished top-two indices;
it need not enumerate 8! permutations. Alternatively label the current operation
as partial symmetrization and assess its numerical effect before relying on it.

### 4. Zero-variance atoms require an adaptation decision

The inherited `wick.py` clamps variances below 1e-10 up to 1e-10. Its Heaviside
coefficient at mean 0, variance 0 is therefore **0.5**, while the exact wrapper's
strict Heaviside is 0. Its ReLU mean at the same deterministic point is
**3.989422804e-6**, instead of 0. These are directly reproduced from the vendor
functions, not inferred from a failed end-to-end run.

This is an inherited nondegenerate-Gaussian regularization assumption, **not a
demonstration of an error in ARC's stated algorithm or random-MLP setting**.
The RNN adaptation introduces degenerate and atom-valued comparison variables;
constant/duplicate comparisons should be simplified or given explicit exact
handling. Initial h0=0 alone does not prove a propagated artifact: the first
linear layer usually gives recurrent units nonzero variance. We did not measure
how often the clamp activates in the trained DEV propagation.

### 5. Raw FLOP totals are primitive-counter proxies

The wrapper correctly requests the vendor's **raw**, undiscounted count rather
than theoretical symmetric-tensor savings. Its one-line assertion-removal patch
affects discounted accounting, not raw accumulation.

Nevertheless, under the shipped counter definitions, on 100 elements:

| Primitive | Reported count |
|---|---:|
| Gaussian CDF |300 |
| erf |0 |
| degree-4 Hermite evaluation |0 |
| exp |100 |

The CDF's surrounding arithmetic is counted while its erf kernel is not.
Transcendentals generally use a primitive operation convention, and some
special functions lack registrations entirely. NumPy setup of model-dependent
layer matrices occurs outside the propagation counter too. Thus “raw” does not
mean all hardware arithmetic or complete shared-harness FLOPs.

Declare this convention, account for unregistered special functions and setup
if producing equal-compute leaderboard claims, and compare CPU time as the
shared rule requires. **The aggregate impact is unquantified:** dense tensor
contractions may dominate. This finding does not invalidate reported method
accuracy or prove that its compute comparison changes sign.

## Reproduction

From this worktree root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/clean-neurons/.venv/bin/python review/arc_cumulants/check.py
```

The script writes `results.json` in this review folder. The checkpoint is read
from the existing main-worktree cache, with SHA256
`498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`.
No external model or dependency is fetched. DEV inputs use seed 42002; synthetic
checks use 42001. No fitting, estimator selection, or leaderboard scoring occurs.

## Compute accounting

Two review runs were made, the second adding exact primitive-cost and ReLU-atom
checks. Each compares direct and layer-stack recurrences on 32,768 DEV inputs
plus 518 synthetic inputs: **66,572 sequence evaluations per run**, 133,144 total.
The repeated layer stack is charged as a full recurrence evaluation even though
it is only validation. There is no 4M cell precomputation, K3/K4 propagation, new
truth sweep, or model training. Recorded script main CPU times were 1.20 and 1.02
seconds, excluding imports; both shell processes together used under 6 seconds.
FLOPs for review arithmetic were not instrumented and are not an estimator cost.

## Claims

- [measured] Exact readout equivalence fails on tied logits, but no such mismatch
  was observed on the two 16k DEV samples.
- [measured] AUGMENT K2 input-degree truncation, subchunk failure, incomplete
  symmetry, atom regularization, and primitive-counter omissions reproduce.
- [inferred] These justify targeted wrapper fixes and scoped accounting notes;
  they do not establish why full cumulant propagation succeeds or fails.

## Alternatives not ruled out

Higher cumulant-order closure error, the Heaviside endpoint's non-Gaussian law,
training-induced dependence, and numerical instability may dominate the actual
M16,T10 error. Full propagation and actual cached conditional moments were not
reviewed numerically. Zero observed ties in 32,768 DEV inputs does not prove
their population probability is zero.

## Provenance and failure modes

The local vendor declaration pins ARC's
`mlp_cumulant_propagation` commit
`93d091a4c26c042bfffa28f2e76a81bc0aba94bb` (2026-08-04), omitting symb/ and
plotting_utils.py, under the included MIT license. It declares one local patch
to flop_utils.py; that marked patch is present. No upstream checkout was
downloaded for bytewise verification, so the declared provenance is not
independently attested. These post-January2026 sources are already logged in
`lit/web_log.md` by the DRI; this audit reads local code only and opens no new
external source. It depends on that vendored implementation.

The audit uses AST extraction because optional dependencies are unavailable;
this verifies the named functions, not a fully installed vendor execution.
Snapshot drift, aggregate-only checks hiding tied-output swaps, degenerate
variance assumptions, and incomplete FLOP registrations are the main risks.
