# DLASQ4 bounded source numeric envelope

Let L=N0-I0+1, 3<=L<=4. Count scalar arithmetic, unary negation, square roots,
min/max and floating comparisons as one named unit each; integer/index/control
and memory remain excluded. Both floating operands of logical conjunctions are
charged even if the compiler short-circuits. For alternatives take the larger
branch, not a sum. Assignments/copies/constants have no named numeric operation.

The source has no calls or unbounded loop. Case4 loop10 executes at most L-2
iterations (or L-3 in its DN1 arm). Case5 loop30 executes at most max(0,L-4),
hence zero here. Cases7/10 loops50/70 execute at most L-2 iterations.

| Source shift case | All-path source upper | Main terms including outer tests |
|---|---:|---|
|1, DMIN<=0|2|comparison + negation|
|2/3|32|1+2+7+2+3+2+4+2+9|
|4|26+9(L-2)|1+2+7+2 +1+1+4+1 +9(L-2)+1+1+5|
|5|20|1+2+1 +1+2+4+1+1+1+1+5|
|6|8|1+2+1 +3+1|
|7/8|24+7(L-2)|1+2+1+1+1+1 +7(L-2)+2+3+2+3+7|
|9|6|1+2+1+1+1|
|10|29+6(L-2)|1+3+1+1+1+1 +6(L-2)+2+3+6+3+7|
|11|5|1+3+1|
|12|1|initial DMIN comparison; remaining control integer|

Thus35 at L3 and44 at L4 bound all source paths. The frozen translation charges
short-circuit evaluated operations and must remain below this conservative table.
Native parity controls check the source interpretation on a finite panel; they do
not prove all-input compiled equivalence. The table is a pinned-source envelope,
not an instruction/cycle bound on every possible compiler implementation.

Q3 entry lengths1/2 return after deflation before calling Q4. For initial lengths
3/4, Q3 calls Q4 at most once after deflation; current length cannot exceed entry
length. Therefore the saved Q3 entry histogram gives a safe source upper using
0/0/35/44 by entry length. No Q5 work is included here, and retries do not repeat
Q4. No generic retry-termination or all-matrix rank cost claim follows.
