# Exact quadrature constants for T5

## Question
Can the ideal Gauss-Legendre nodes and weights underlying the fixed T5 positive rules be enclosed exactly, and can their differences from source binary constants be bounded?

## Method
Claim T11588 ba23ebd1. Rational Legendre/Sturm isolation to160 dyadic bits, interval derivative weight formula, outward128-bit weight endpoints. Orders6,10,16 only; fixed arithmetic/root controls and independent source gate precede target execution. See PROOF.md, exact.py and protocol.json.

## Status
All13 controls and fixed n6/10/16 target run pass after reciprocal source gates. Independent saved audit d5fc9359 checks all32 nodes/weights and64 moments via closed Rodrigues coefficients and original endpoint signs, without source-generation or bisection reruns.

## Results
[measured] All32 ideal node/weight intervals are bounded; no unresolved cases. Source component errors are at most approximately8.878e-17 for nodes and4.364e-16 for weights (exact rational intervals authoritative). Target CPU .624518240 seconds. [inferred] Exact root isolation plus weight interval arithmetic can certify the ideal constants; [speculative] their source deviations may be small enough for a later integrand perturbation bound.

## Compute accounting
One thread,2GB, review lock20CPU per cycle. Rational/integer operations and peak operand sizes retained; model-independent setup separated. No tensor, model, mass, geometry or MC evaluations. Exact setup is not called free runtime work, and no ranked entry is produced.

## Claims
Only ideal node/weight inclusion and source-distance diagnostics are in scope. No whole-integral/reference accuracy or native speed claim.

## Alternatives not ruled out
A root or derivative enclosure may remain unresolved at fixed precision; retain it rather than increase precision after viewing results. Accurate nodes may still be insufficient because the integrand is ill-conditioned; a separate perturbation bound is required.

## Failure modes
Miscounting roots at zero polynomial entries, treating moment inclusion as uniqueness, rounding reciprocal endpoints inward, erasing unknown denominators, pricing exact arithmetic as one cheap FLOP, or transferring component accuracy to the full source integral.

Independent theorem gate T55 and source/control gate recover34 pass. Named square extremum count2 conservatively covers its zero-crossing branch (one actual max comparison). Integer/GCD/runtime and transient unreduced Fraction widths remain excluded; saved peak bits concern retained reduced rationals.
