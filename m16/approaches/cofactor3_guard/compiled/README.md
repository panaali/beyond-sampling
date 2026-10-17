# Compiled guarded cofactor gate

## Question
Does a strictly compiled fast kernel preserve the existing reviewed enclosure and what arithmetic does it execute?

## Method
PROTOCOL.md freezes existing25+64 controls, build flags, fenv rejection and explicit fallback ownership. guard.c ports the separate binary64 trees. build.py saves compiler/source/library/disassembly pins without executing the kernel.

## Status
Running; claimf058ce6f. Source gate pending. No kernel execution or numerical result yet.

## Results
Pending. The existing Python theorem and exact control reviews cb751402/fd0dabc7 remain prerequisites.

## Compute accounting
Zero model/mass/MC/native determinant calls. Build and finite-control costs will be recorded separately. External nextafter/fenv, interpreter/control/memory and exact fallback costs are not automatically priced.

## Claims
- [measured] Fixed source/control scope frozen before build/testing.
- [inferred] Bit parity plus strict assembly can support applicability of the reviewed binary64 theorem under explicit runtime assumptions.
- [speculative] A compiled enclosure could reduce primitive overhead; no speed or total-mass claim yet.

## Alternatives not ruled out
Caller environment changes, library behavior, cheaper native paths, fallback frequency and downstream interval propagation remain open. The104 native envelope is not a lower bound.

## Failure modes
Fast-math/reassociation/fusion, unsupported rounding/FTZ, undocumented fallback and ignored library work can invalidate stronger conclusions. Existing point cancellations are retained; enclosure is not an accurate-point acceptance test.
