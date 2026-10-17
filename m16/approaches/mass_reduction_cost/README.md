# T11579 fixed mass norm and reduction coverage

## Question
What work do the unchanged four-dimensional mass kernel's norms and sums actually execute, beyond their inherited expression counts?

## Method
Fresh TASKS-only claim e3e2f164 precedes this folder. Pin139a7b9e source and independently reviewed35,098-cone branch census, prior matrix replacement ledger, installed NumPy2.5.4/CPython3.12.14 and public matching source revisions. Inspect length4 scalar norm→dot, row4 and length128/1000 array sums, and six-Python-float built-in sum. Freeze controls and independent source gate before any numerical execution. Replace only matching old terms; do not rerun mass or geometry.

## Status
Source and dispatch inventory in progress. No numerical controls or corpus contraction yet.

## Results
Read-only source inspection shows scalar norm uses x.dot(x), not the previously priced DNRM2. NumPy addition reductions use a pairwise helper with extra accumulator work for short lengths. CPython3.12 sum has compensated exact-float branches, so its inherited five additions are not complete source work. Actual dispatch/layout/control gates remain open.

## Compute accounting
No model, mass, geometry or random-input evaluations. One thread,2GB, review<=20CPU, primary<=55CPU. Numeric categories and remaining integer/control/memory/libm/runtime qualifications will remain explicit; no sampler multiplier or full native-cost claim.

## Claims
[inferred] Source-path differences can change the inherited expression subtotal. [speculative] Fixed-layout controls and pinned native code can bound these particular categories.

## Alternatives not ruled out
Different reduction initialization/chunking, different native dot dispatch, unexpected scalar types, SIMD lane work and historical binding differences. Cheapest discriminator is a frozen source/native/layout panel including signed zero and cancellation, followed by only saved-count contraction.

## Failure modes
Confusing source categories with executed SIMD lanes; overlooking pairwise initialization or compensation; assuming NumPy norm means DNRM2; applying contiguous costs to strided inputs; double-counting prior matrix terms; treating finite parity as all-input implementation proof. sysconfig LIBDIR=/install/lib does not locate the actual static Python runtime; preserve this metadata-path mismatch and pin the executable itself.
