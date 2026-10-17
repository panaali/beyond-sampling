# DLASQ5 named numeric source envelope

This is a bound on the pinned source's floating arithmetic, negations and numeric comparisons/MIN, not all native rank work or hardware instructions. Integer index arithmetic, loop/dimension/IEEE/PP control, memory and runtime remain separate. The finite installed controls support source relevance; they do not establish whole-build equivalence for arbitrary values.

Let L=N0−I0+1≥3. The explicit main loop has max(0,L−3) iterations. Either parity executes the same count. Both shifted and zero-shift versions have the same arithmetic; the latter adds a possible threshold comparison in each main-loop body. The non-IEEE branch adds a possible negative-D early return. Early returns can only reduce the counts below the complete-path envelope.

| Piece | Arithmetic (+,−,×,÷) | Negation | Numeric comparison/MIN |
|---|---:|---:|---:|
| Prologue: EPS*(SIGMA+TAU), threshold*HALF, initial Z−TAU; DMIN1=−Z |4|1|2|
| One IEEE main-loop body, zero-shift envelope |5|0|3|
| One non-IEEE main-loop body, zero-shift envelope |6|0|4|
| Each of two IEEE unrolled tails |6|0|1|
| Each of two non-IEEE unrolled tails |6|0|2|

Prologue comparisons are TAU<threshold*HALF and TAU!=0. The main-loop IEEE body has one sum, one quotient, multiply/subtract D, one output multiply, two MIN calls and at most one threshold comparison. Non-IEEE recomputes two quotients/two products plus the sum/subtract, and tests D<0 before continuing. Each tail uses sum, two products, two quotients, subtraction, MIN and, in non-IEEE mode, one guard. Final outputs are assignments only.

Therefore all branches have componentwise upper envelope `(6L−2 arithmetic,1 negation,4L−6 comparisons)`, total `10L−7`. IEEE-only gives `(5L+1,1,3L−5)`, total `8L−3`. At L=3/4 the conservative totals are23/33; IEEE-only21/29. Do not use the tighter count merely because the platform is normally IEEE: the saved histogram does not record that flag or the effective shift.

The census observer explicitly records `m=*n0-*i0+1`, `n=*pp` for Q5. Its actual L3/4 counts can thus be multiplied by23/33. This uses observed retry calls and never assumes a universal Q3/Q2 convergence bound. No Q6 calls were observed in these fixed source executions. Q3-exclusive work, shift selection, Q2/Q1 work, reductions, reflectors, BLAS, setup and wrappers remain unpriced. A future full parent price must exclude this Q5 subtotal or explicitly retain double counting only as a proven conservative upper bound.

## Frozen finite controls
64 cases: L3/4, I0=1/2, PP0/1, TAU0/.125/4/2^-60, IEEE false/true, SIGMA=.5, EPS2^-52, Z[j]=1+j/8. These include shift suppression and non-IEEE early returns. The counted direct scalar translation matches every pinned installed output and updated TAU bitwise, and every category is below its envelope. Maximum counts reach all four stated23/33/21/29 totals, so no observed branch was silently assigned a lower count. CPU1.067303, peak57792KiB; repeated library hashing is validation work, not candidate advice. No real-model matrices, Gaussian mass or geometry were reevaluated.
