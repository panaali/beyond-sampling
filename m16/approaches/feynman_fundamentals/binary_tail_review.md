# Binary64 positive tail review

T55 read-only review of T78 `binary.py` through c767438, 2026-10-17.

The q^-3 polynomial is the same as the independently checked Decimal formula: `N^2+2+3*N*alpha+3*z*(z+2)`. All remaining terms and Eulerian geometric sums are nonnegative. The declared degree restriction keeps N^2+2 and the other integer constants exactly representable in binary64. Integer powers use explicit outward multiplications.

Assuming correctly rounded binary64 basic operations and square root, stepping each elementary lower result down and upper result up encloses it. Lower clipping at zero is valid because these analytic intermediate quantities are nonnegative. In particular, subtraction is only used for b-a and 1-r. A denominator interval reaching zero loses finite upper information. NaN intermediates become [0,infinity], avoiding silent invalid finite results. Overflow and underflow can therefore produce very loose intervals. Equal radial endpoints and zero determinant are restored to their exact zero bound after evaluation. Inputs are exact stored binary64 endpoints; this does not certify how they were obtained.

Pairwise sum performs outward additions at every internal reduction node, carrying an odd leftover endpoint unchanged. It avoids an ordinary floating sum. Empty input returns zero. Decimal Machin pi setup is included per bound call; this and Fraction setup remain separately charged kernels, not a portable instruction audit.

`check_binary_tail_review.py` uses a separate 200-digit closed q^-3 expression for eight fixed cases: constant, adjacent endpoints, tiny/huge scale, wide interval, ordinary low order, underflowing power, and the largest allowed degree. Every returned interval contains the reference. Two scale extremes correctly return [0,infinity]; the two enormous-degree cases have conservative positive upper bounds instead of silent zero. CPU was 0.00660 seconds. This check exercises bounds, not original cone geometry or mass integration. The initial invocation had a local script-path mistake, fixed before the numeric run; no numerical failure was discarded.

The cost measurement on tiled sixteen-cone inputs is an arithmetic-overhead benchmark, not 4096 independent geometric cases or a full planner timing. A geometry-certified pipeline still needs outward radial/determinant inputs and its numerical integral enclosure.
