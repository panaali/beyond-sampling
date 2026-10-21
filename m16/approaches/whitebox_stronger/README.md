# T475 deadline stronger public baseline reproduction

## Question
Can an immutable public stronger estimator improve our first submission before the owner deadline?

## Method
Run unchanged pinned e30_a, logxio and V29 in isolated supervised workers, sequentially and single-threaded. Sources and hashes in sources.json. Compare predictions to the same official archived public means used by T474. First two mini rows are development; full rows are held out only from our development, not from upstream authors.

Two alternatives: published improvements may come from a different scorer version (test actual flopscope cost and residual); learned public corrections may overfit (compare distinct full rows, retain limitation). Stop any worker above 7.5 GB RSS, 120 CPU s or150 wall s. Root reports owner authorized 8 GB/process and16 GBtotal for this deadline sprint. Maximum8 CPU minutes total.

## Status
Running.

## Results
Artifacts in results; root registers final numbers in INDEX.

## Compute accounting
Actual flopscope operations; setup separately metered. Supervisor wall/RSS/CPU counters retained. Four reference networks maximum per promising source.

## Claims
[inferred] Published methods are likely stronger than V25; local reproduction required. No novelty, private rank or broad generalization claim.

## Alternatives not ruled out
Public training overlap, floating-point platform effects, grader-version changes.

## Failure modes
Residual time .4s, full shape1024x16 only, memory > limit, numerical stability, learned corrections.
