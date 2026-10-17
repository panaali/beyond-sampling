# Frozen primitive controls

Source gate before execution. No data-dependent fixture selection or new model evaluations. Binary64 inputs are constructed by exact powers of two and the listed stored constants. Order is fixed:

1. All positive zeros.
2. All negative zeros.
3. Identity.
4. Diagonal(-1,2,3).
5. Cyclic permutation[[0,1,0],[0,0,1],[1,0,0]].
6. Dense integer[[1,2,3],[0,4,5],[1,0,6]].
7. Duplicate rows[[1,2,3],[1,2,3],[4,5,6]].
8. All ones.
9–11. A(delta)=[[1,1,1],[1,1+delta,1],[1,1,1+delta]], delta=2^-50,2^-40,2^-26.
12. Diagonal(2^100,2^100,2^100).
13. Diagonal(2^-100,2^-100,2^-100).
14. Diagonal(2^-100,1,2^100).
15. 2^99 A(2^-50).
16. 2^-99 A(2^-50).
17. Identity with first diagonal nextafter(2^-100,0), outside the lower guard.
18. Identity with first diagonal nextafter(2^100,+infinity), outside the upper guard.
19. Identity with first diagonal the smallest positive subnormal.
20. Diagonal(1e308,1e308,1e308), finite inputs but determinant overflows binary64.
21. Diagonal(1e-200,1e-200,1e-200), exact determinant below the subnormal range.
22–24. Identity with first diagonal NaN,+infinity,-infinity.
25. Identity with every off-diagonal entry negative zero.

For every finite case, independently compute the exact stored-input determinant using a six-permutation rational sum. Check enclosure containment with rational comparisons against finite endpoints; infinities are treated by their ordering. Preserve all raw point values/errors, exact numerator/denominator, interval endpoints, guard/fallback status, source-category counts and maximum fallback operand bit lengths. Fast zero cases must remain distinguishable from sign-resolved nonzero cases. Nonfinite cases must return unknown without exact arithmetic or a fabricated real-value enclosure.

Do not count an enclosure as a successful relative estimate merely because it contains the truth. Record whether the interval includes zero and whether the raw fast point equals zero despite nonzero exact determinant. No throughput timing comparison or mass-level claim. Independent results review is required before selecting an existing frozen cofactor sample.
