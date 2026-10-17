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
