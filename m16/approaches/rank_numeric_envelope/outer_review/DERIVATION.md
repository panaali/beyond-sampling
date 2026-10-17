# Candidate outer bidiagonal source envelope (not yet contracted)

## Preconditions and unresolved call graph
All pricing below requires original Q1/Q2 dimension N<=4 and INFO=0. LastN
observer fields are not a maximum. Cold eigensolver exclusion is being checked
separately; until independently closed, do not apply the global Q1/Q2 counts.
Existing Q3/Q4/Q5 row-based prices are unaffected. Source numeric units exclude
integer/index/character control, memory/copies, casts and runtime; no compiled
all-input instruction equivalence is asserted.

## Small helpers
DLAS2 bound28: common3 abs+2 min/max+1 comparison=6. Zero-min branch adds
1 comparison and at most8 formula units (15total). Otherwise GA<FHMX comparison
makes7; smaller-G path AS/AT/AU6 +C8 +outputs2 =>23. Larger-G path AU/guard2,
AS/AT4,C10,SSMIN3,SSMAX2 yields28. Early AU=0 reduces this.
DLASRT N<=4 uses insertion sort, at mostN(N-1)/2 floating comparisons and swaps
only. Equal values do not swap; character validation is unpriced.
IEEECK source uses REAL (binary32 in this installation): infinity checks have
10 arithmetic/negation +8 comparisons=18; NaN branch adds6 arithmetic and
6 comparisons=12, so30. Installed code optimizes some repeated operations away;
retain source30. ILAENV ISPEC10 invokes IEEECK1; no other floating work on that
selector path. DLAMCH floating work remains zero with character control unpriced.

DLASCL TYPE=G positive finite CFROM/CTO: source validation costs3 numeric
comparisons (CFROM zero, two NaN checks); BIGNUM=1/SMLNUM costs1. Each scale
iteration costs at most13 source scalar units plusK element multiplies. Installed
SMLNUM=2^-1022 and BIGNUM=2^1022. Each nonfinal step scales only the larger-magnitude
side down by2^1022; the test ensures it remains larger, so direction cannot
alternate. After at most two such steps, another product/division by2^1022
underflows to0 (initial finite magnitude<2^1024), which cannot pass the nonfinal
strict inequality against a positive counterpart. Thus at most3 iterations,
including final division. Bound43+3K. Early MUL=1 return only lowers it. This
requires ordinary binary64 round-to-nearest/gradual underflow behavior and pinned
constants; controls will exercise extreme finite endpoints. Zero/NaN/Inf ratios
are outside this positive finite proof. Q1's SIGMX/SCALE ratios meet this domain
when its offdiagonal is nonzero and data finite.

## Q1
N1 costs1 abs; N2 calls DLAS2<=28. For N3/4, exclusive normal path costs7N:
3(N-1)+1 initial abs/max, zero test1, max-loopN, scale divide/sqrt2,
2N-1 squarings andN output square roots. Early diagonal path plus small sort
is less than28. Thus28 per Q1 covers either exclusive path or its N2/diagonal
helper, with no duplicated DLAS2/sort price. Each actual Q2 call additionally
covers Q1's two scales: K<=7 andK<=4, bounds64+55=119. Q2 itself is separate.
DCOPY is memory-only in these named units, not zero runtime cost.

## Q2 success path
For N3/4 initialization exclusive cost11+44(N-1): setup2, data scan8(N-1),
last item4, diagonality1, trace sum/test2, reverse test2; two dqd sweeps each
have backward5(N-1), forward12(N-1), and maximum-loop(N-1).
Each of at mostN+1 outer iterations costs<=12N outside Q3: sigma2, initial
abs1, unreduced scan8(N-1), minimum-ratio scan4(N-1), final test2 and initial
shift7. Array reversals are copies, integer/index work unpriced. Successful
final sum plus diagnostics costsN+3. Add sort<=6 and IEEECK<=30. At N4 this
is143+240+7+6+30=426, safely also boundingN1/2 andN3 success branches.
After each Q3 call, split detection costs at most11 only when entry L>=4:
outer floating test4, at mostone loop iteration at L4 costing4+max(1,3)=7.
Current active length cannot exceed entry length. Use actual Q3 L4 histogram.
INFO2 cleanup is explicitly excluded by observed successful Q2 calls; its
previous source-only nonprogress witness is not dismissed or repaired here.

If all premises and helper/source controls pass, the proposed nonoverlapping
outer subtotal is28*Q1calls +(119+426)*Q2calls +11*Q3entryL4calls.
This excludes already priced Q3/Q4/Q5, reflectors, and all matrix-reduction work.
