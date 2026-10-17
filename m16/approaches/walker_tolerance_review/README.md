# T30 legacy walker tolerance and completion review

## Question
Do exact-tie, tiny-gap, close-root and iteration-cap cases match the advertised
segment-integral semantics? Which interfaces explicitly report a remainder,
and which callers discard one? Parent reserved T30 in2aaba6b. All author/shared
sources are read-only; local review artifacts live here.

## Method: frozen before runs
Build an independent small reference by propagating exact binary-rational
affine hidden maps over a line. Split every gate at its exact rational zero,
then split at all exact readout pair intersections and evaluate the first
argmax at each rational midpoint. No numerical root tolerance, lookahead or
gate-probe epsilon. Integrate the resulting intervals with high-precision
Gaussian CDFs; geometry is exact for the supplied binary64 parameters/line.
This is a small reference and strongest correctness repair, not a fast
production walker. A cap must raise rather than silently return partial mass.

Fixed fixtures (T10,H1/2, tau=.25, no trained-model/noise sweep):
1. All-zero logits with pair0/1: first-index prediction0 implies exactly half
   the symmetric segment is wrong; fixed-target1 tube side is entirely wrong.
2. Reset RNN with constant positive final input and only logit1 positive:
   output scales1,1e-14,1e-16 leave truth unchanged. Positive gaps are not ties.
3. Reset RNN with final input -t/sqrt2: a gate crosses at0; prediction9 is
   correct for t<0 and zero-state prediction0 is wrong for t>0 with pair8/9.
   Check max_cells/max_iter0 and1 against the full integral. Tube K0/1 must
   expose its unwalked mass rather than be called incorrect merely for stopping.
4. A two-state positive affine model gives readout envelope0→9→8, with two
   crossings near-.125 separated by1e-9 (and control separation1e-5).
   Pair8/9 makes the middle interval correct. Fixed1e-7 readout lookahead may
   skip that interval. Retain exact binary-rational oracle crossings and test
   output scaling. Also check wtool maxseg1 for declared failure or truncation.

Read-only targets: sci_tao/core.py, sci_feynman/core.py and wtool.py,
bandwidth/wtool.py, boundary/zones.py, tube/tube.py and estimator.py,
sci_laplace/lqt.py. Run callable small cases for the first group and source-audit
wrapper remainder handling. Import modules by absolute path without changing
their globals. Existing corrected sci_tao_v2 may be compared if callable.

Alternative explanations and cheapest discriminators: a near-root mismatch
may be CDF cancellation (compare70-digit interval mass) or wrong interval
topology (compare rational winner intervals). A cap shortfall may be declared
approximation (inspect returned remainder) or a silent completion bug (only a
scalar mass returned). Exact score ties and strictly positive tiny gaps are
different controls. Keep all failures; do not alter source to hide them.

## Status
Protocol frozen; reference and bounded fixtures next. Numerical cycle<=110CPU,
2GB, primary shared lock, single BLAS/OpenMP thread. No family-wide negative
claim and no claim about how often historical trained-model runs hit a defect.

## Results and reproduction
Pending. Exact commands and hashes will be recorded with results.

## Compute accounting
Count reference affine products, exact gate/readout roots, final intervals,
high-precision mass calls, CPU and RSS, plus tested walker calls. Count setup;
no audited FPE efficiency claim for the Fraction reference.

## Claims
[inferred] Exact rational line geometry removes epsilon/lookahead ambiguity for
the finite fixture set. Numerical CDF evaluation remains separate from topology.
[speculative] Local corrections may fix the observed defects without changing
the broader face-integration method; production performance remains untested.

## Alternatives not ruled out
Adaptive tolerances, exact event arithmetic, explicit completion/remainder
interfaces and finite-K approximations with honest uncertainty are viable.
This audit does not decide their production cost or accuracy on all checkpoints.

## Failure modes
Exact rational growth can be expensive. Floating runtime semantics differ
from exact-real interpretation of binary parameters. Legacy modules hardcode
T10, so use T10 fixtures consistently. A source audit of ignored remainder does
not measure its frequency or magnitude in past experiments.

## Follow-up frozen after first fixtures
Keep the first results unchanged. Compare both corrected sci_tao_v2 Python and
available compiled backend on all nine fixtures, including cap0/1 failures.
Record source/binary hashes. These are separately labeled existing corrections,
not legacy methods or a new independent oracle. Local in-memory repairs only:
replace legacy Tao's tolerance tie with c==0 and raise if active lines remain;
replace tube's fixed-target zero-margin test with first-index argmax semantics.
No change to eps/gate advancement is made, so success does not prove all near
gate roots handled. Exact rational reference remains the strongest topology
control for these fixtures. Follow-up<=15CPU, same resource limits.

## Results: finite implementation defects, not method rejection
Both bounded runs completed: first0.57470CPU seconds including module setup
.54776; corrections .38431CPU seconds; total .95901CPU seconds, excluding
interpreter imports/report serialization. Peak first-run RSS123532KiB. No random
model-input samples or trained-model benchmarks were used. Every source hash,
exact reference interval, scalar call and reported work count is retained.

| Fixture | Reference wrong Gaussian mass | Measured legacy behavior |
|---|---:|---|
| All-zero logits, pair0/1 | .09870632568292373 | Tao/Feynman/wtool/zones agree within2.78e-17 |
| Positive logit1 at scale1 | .09870632568292373 | All four agree |
| Same at1e-14 or1e-16 | .09870632568292373 | Legacy Tao returns .1974126513658474; other three agree |
| Gate at0, full walk | .09870632568292373 | All four agree |
| Same, cap1 | .09870632568292373 | All four return0 without a completion flag/remainder |
| Output crossings1e-5 apart | .09870236730358037 | All four agree |
| Output crossings1e-9 apart | .09870632528708600 | Feynman/wtool/zones return .09870632568292370; Tao agrees |

The close-root error3.95838e-10 is small in absolute probability but loses the
whole narrow correct interval. Scaling all output weights by1e-16 preserves the
failure. This is topology/lookahead error, not CDF subtraction: the independent
reference uses exact rational endpoints and70-digit CDF evaluation.

Cap0 probes also return0 silently, but these legacy docstrings do not explicitly
promise cap0 support. The cap1 fixture is decisive without that edge convention:
one valid gate cell is processed and a positive-mass second cell is omitted.
The functions describe exact mass, return ordinary outputs, and expose neither
completion status nor remaining mass. This does not establish that their large
default limits were reached in historical experiments. wtool maxseg1 raises an
IndexError because its segment counter exceeds storage capacity; this is a
failure, not a silently valid truncated estimate. The bandwidth/wtool.py source
is byte-identical to sci_feynman/wtool.py at this review.

Tube's finite-K contract is different and should be preserved. On the gate
fixture with a fixed always-wrong target, K0 returns error0/remainder.19741265;
K1 returns error.09870633/remainder.09870633. Those are honest partial integrals.
But on an all-zero model with fixed target1, K1 returns error0/remainder0 although
the whole side is wrong: a separate first-index tie bug. The local tie repair
returns error.09870633/remainder0 and preserves the K0 remainder.

## Strongest plausible corrections checked
The existing corrected **sci_tao_v2 Python** implementation agrees with all nine
reference fixtures to<=2.78e-17 and raises on cap0/1. Therefore legacy failures
must not be attributed to newer results/costs using this corrected walker.
Its exact source SHA is
`aea5f9008fe9cc8a5436226f1057b5e0527f119d832df536127d8370bc7968f9`.
The compiled wrapper/source were hashed but its shared-library binary is absent
from this checkout; calls explicitly fail "C++ backend absent". No compiled
comparison is claimed and no shared binary was built.

Local in-memory Tao fixes (exact zero-gap tie test, explicit incomplete-walk
failure) also pass all nine fixtures and cap tests. The tube first-index repair
fixes the constant-target control. These leave gate eps/probes unchanged and
are not a general near-degenerate-root certificate. The Fraction reference
eliminates such ambiguity for these small cases and explicitly raises on its
own piece cap; it has no production efficiency claim.

T55 independently read rational_line.py and fixture construction and verified
the exact-affine/gate/all-pair splitting argument, first-index midpoint rule and
both closed-form controls. For the reset model h_final=1, positive logit1 wins
at every tested scale and the wrong interval is(0,.25). For the narrow envelope,
only(r1,r2) on the negative half is correct, giving
Phi(0)-Phi(-.25)-[Phi(r2)-Phi(r1)]. Actual dyadic roots are used, not intended
decimal roots. This is independent reference/source review, not an independent
rerun of every legacy module, and not interval certification of special functions.

## DRI handoff: source fixes and remainder flow
No shared files were changed. Suggested follow-ups for maintainers:

* Legacy sci_tao/core.py: replace approximate ties in zero-slope comparisons
  with exact-zero semantics or a validated error enclosure; raise/return a
  remainder on unfinished cells. Existing v2 already handles these fixtures.
* sci_feynman/core.py, both wtool copies and boundary/zones.py: replace fixed
  readout lookahead with a complete affine upper-envelope/correct-set solve.
  Bound or reject unprocessed cells/output pieces; enforce capacity before
  incrementing segment counters. Never turn a tiny positive gap into a tie.
* tube.walk_side: add first-index ownership of identically equal logit rows;
  retain its useful explicit remainder interface. An error interval is
  [computed_error,computed_error+remainder] only after processed-cell correctness
  is established; the tie bug shows why remainder alone is insufficient.
* tube.est_tube and est_tube_newton discard the F_faces remainder (`F,_,C`
  and `Fe,_,Ce`). Propagate it or fail when it exceeds a declared bound.
  tube.F_hybrid similarly discards the near-part remainder from its K200 walk.
  Its far-part one-node integration is an explicitly documented approximation,
  not an exact-walker defect. sci_laplace/lqt.py calls this hybrid; lp.F_sides
  separately drops both side remainders. Do not label all their tail
  approximation error as silent walker truncation.

Status: bounded independent fixture review and correction check complete.
These counterexamples invalidate unconditional exactness descriptions for the
listed legacy implementations. They do not establish the size/frequency of
errors in archived trained-model evaluations or reject face integration itself.
Untested variants: adaptive certified gate stepping, exact production event
arithmetic, complete capacity stress at default limits, and all trained-model
frequencies. They remain open because this task targets small decisive fixtures.

Reproduce from the worktree root with the common memory/thread/lock prefix:
`/workspace/clean-neurons/.venv/bin/python m16/approaches/walker_tolerance_review/check.py`
then `corrections.py` in the same directory. `results.json` preserves original
failures; `corrections.json` distinguishes local adapters, v2 Python success and
unavailable compiled control. Research commits should be integrated by root;
only the DRI registers or confirms project-wide claims.
