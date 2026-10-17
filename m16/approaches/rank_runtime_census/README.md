# T165: frozen rank-kernel runtime census

## Question
Can the actual finite candidate evaluation resolve the unknown native SVD
iteration work without assuming a universal convergence/failure-path bound?

## Method
TASKS-only claim cb6ed1c0 precedes implementation. Pin T156 binaries and
pass-through observer, T146 unchanged source loader, T153 recipe capture,
and saved seed0 model/rays. Extend the observer with bounded kernel-call
rows keyed by kernel, dimensions and workspace-query status. Forward every
argument and return value unchanged. Count actual DGESDD, QR/LQ/bidiagonal
reductions, reflector/norm work and dqds calls/iteration deltas. Preserve
unobserved failure branches and missing native categories explicitly.

First reuse the saved analytic rank and tridiagonal controls, comparing all
outputs bitwise. Also compare zero/rational toy geometry baseline to observed
geometry with every ordered ray and source counter fixed. Independent source
review precedes trained execution. Then exactly eight twelve-parent chunks
per mode, exact and recipe108, with a zero mass callback that validates every
ray against T131/T145 arrays. No probability integration or candidate tuning.
Cold setup is saved separately and composed once. Native call/iteration counts
are evidence about these two fixed evaluations, not a portable worst-case
cost bound or hardware count. Numeric operation envelopes still need source
analysis and independent review.

## Status
Protocol frozen; pass-through counter implementation next.

## Results
Pending. T156 failed workspace diagnostic remains preserved in its own folder.

## Compute accounting
One thread, 2 GB, <=20 CPU seconds review controls, <=110 CPU per trained
chunk under the primary lock. All experiment/review CPU is reported separately
from estimator operation accounting. No Gaussian quadrature or model sampling.

## Claims
[inferred] Actual native iteration counters can replace assumptions about
unreached failure paths for these finite evaluations.
[speculative] Kernel census will make a defensible numeric budget practical.

## Alternatives not ruled out
Source-only cap after T163 proof, replacement rank methods, platform-dependent
kernels, uninstrumented native helper work, different candidate/model panels.

## Failure modes
ABI mismatch, missed internal bindings, changed ordered rays, incorrect
workspace-query categorization, counter overflow, setup duplication, treating
an observed path as a universal bound, or conflating timing with arithmetic.
These are discriminated by pinned binaries, known analytic controls, exact
baseline outputs, source call sites, and saved geometry/counter parity.

Initial generator had a Python f-string brace typo before any native build;
it was corrected before controls. All 114 analytic and three tridiagonal
native controls then passed bitwise, including Q5+Q6=ITER increments. Four
toy baseline geometries were saved. The first observed-geometry control
incorrectly required every toy (including the zero network) to execute a
rank call; the zero network has none in the geometry phase. Its assertion
failure is preserved in failure_observed.json. Require the rational toys
to exercise rank, while all cases retain exact ordered-ray/source parity.
Source inspection also found the prior ledger key is opaque_calls, not
opaque; fix this before any trained execution. These are control/adapter
issues, not numerical estimator failures.

Corrected toy control passes all four exact/recipe zero/rational geometries,
with bitwise ordered-ray parity and unchanged legacy counts/cell statistics.
The rational models exercise native rank; zero geometry is correctly allowed
to bypass it. Baseline0.419806CPU, observed0.451832CPU; analytic controls
0.0101045CPU. Native Q5+Q6 calls equal Q3 ITER increments in every control.
Source-review gate remains before trained batches.

All sixteen trained chunks completed with exact ordered-ray, cell, legacy
counter and per-chunk rank-call parity. Initial aggregate assertion compared
geometry-only calls to the earlier cold-inclusive rank total, omitting768
setup ranks (576 shape2x4 and192 shape4x4). The failed contraction is
preserved in summary_failure.json. Include one cold setup in the aggregate;
all eight diagnostic setups are retained only in measured review CPU. No
trained rerun is needed.

### Completed finite census
All16 chunks preserve35,098 exact and28,000 recipe rays,8,455/6,182 cells,
and35,650,854/36,398,526 legacy geometry counters. Geometry plus one cold
setup gives275,897/257,040 rank calls, exactly reproducing T146/T153 shapes.
Actual Q3 calls are239,577/200,618 and Q5 updates201,773/165,964; no Q6,
Q1/Q2 failure, or DGESDD failure occurred. Maximum Q3 ITER increment is3
on each finite evaluation. This is observed work, not a universal loop cap.
All reductions use the unblocked QR/LQ and bidiagonal paths represented in
the saved kernel rows. DLARFG counts1,390,372/1,293,660 and DNRM2 counts
886,771/820,870 expose additional native numeric work still needing prices.
Review CPU18.337895/18.810851 seconds, peak RSS205572/204552KiB.

`python summarize.py` contracts existing raw rows only. To reproduce a
trained chunk (after compiling spy.c), run one-thread
`LD_PRELOAD=$PWD/spy.so python run.py exact 0` or `recipe108 0`, indices0..7,
with the protocol lock/resource caps. Source gate T55 review9558b53e passed
before all trained chunks. Final independent aggregate review is requested.
Parent/child kernels are nested: assigning both a full parent price and all
child prices would duplicate work unless explicitly used as an upper bound.
The current output is a call census, with no guessed operation conversion.
No complete-cost, fullpanel, positivity or sampling-win claim follows.
