# Candidate Q3 exclusive source ledger

Scope: saved entry lengths1..4; all observed Q6 call counts zero. This is a
pinned-source named numeric envelope with historical source/binding premises,
not a universal termination or native instruction bound. Q4/Q5 child arithmetic
is excluded. DISNAN's one floating comparison is included; DLAMCH's installed
constant return has no floating arithmetic, with character control unpriced.

Setup TOL=EPS*100 and TOL2=TOL**2 costs2.

The first deflation test costs at most5 (one sum, two multiplies, two comparisons).
A single-eigenvalue deflation adds1. The second test costs at most4. Block40
costs at most23: swap comparison1; T formation3; conditional test3; S formation2;
S<=T comparison1; either revisedS formula7; T update2; product update2; two
sigma additions2. Each successful deflation decreases L by1 or2; if neither
holds the loop exits to50. Therefore F(0)=0,F(1)=1,F(2)=23 and
F(L)=max(6+F(L-1),32+F(L-2),9) for L>=3 gives F3=33,F4=55.
This counts worst-case conditions even if short-circuited.

Reverse block50 costs at most11: DMIN<=0 comparison1, bias multiply/comparison2,
minimum1, two three-way minima2+2, maximum2, unary negative zero1. Array swaps
have no named floating operations; their integer/index/memory work is unpriced.
The compensated sigma accumulation at90 costs one comparison plus4 arithmetic=5.

Each Q5 return's status chain costs at most16: success test2; hidden-convergence
condition8 (2 initial comparisons, sum/multiply/comparison3, abs/multiply/
comparison3); DMIN<0 comparison1; then DMIN1>0 comparison1 and late shift
update4. Other outcomes are bounded by this total, including the one DISNAN
comparison and tau==0 comparison. Integer TTYPE/NFAIL/ITER/NDIV work is excluded.
No universal number of attempts is assumed: multiply16 by the saved actual Q5
call count. Retries return to70 and do not rerun setup/reversal/Q4.

EntryL1/2 cannot reach50, so base2+F is3/25. EntryL3/4 has conservative base
2+F+11+5=51/73. This can overcount mutually exclusive deflation/reversal paths,
which is harmless for a source upper. Add16 per actual Q5 call. No Q6 arithmetic
is priced: the contraction must assert saved Q6=0. A future nonzero row requires
separate helper pricing. No complete SVD/rank/native/runtime claim follows.
