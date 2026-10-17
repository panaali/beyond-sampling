# Remaining driver candidate, not yet contracted

Domain: saved GESDD JOBZ=N, min(m,n)<=4,max(m,n)<=24, successful finite matrices.
No claim for failure/QR fallback, fullvector SVD or larger cold eigensolvers.
Existing children are excluded. Character/index/control/copies/casts remain
unpriced; named source units differ from complete executed runtime.

DLANGE norm M: at most3mn scalar units, one abs and two comparisons per element
including DISNAN. DLARTG: count sqrt/divide constants3, abs2, g/fzero tests2,
range tests4, scaled branch14 (min/max3,divisions2,squares/add/sqrt4,abs/div2,
sign1,division1,final multiply1), giving25. Ordinary branch7 instead of14.
SIGN counts one named operation. Fixed binary constants and source control panel
must be checked before acceptance. DLASCL positivefinite bound43+3K already
independently checked. DROUNDUP_LWORK is unprefixed, source in INSTALL rather
than SRC; wrong path/symbol probes are preserved. Its source upper2 (epsilon
addition and multiply) avoids relying on exact workspace representability;
conversion and integer compare remain explicitly excluded. Public source and
native disassembly agree on this conditional correction.

DGEQRF/DGELQF/DGEBRD choose NB through ILAENV ISPEC1. Need frozen native helper
queries for every saved call shape to prove NB>=min(m,n), so blocked BLAS
branches are absent. Under that condition parents have zero exclusive named
floating arithmetic (workspace conversion/copies and integer logic excluded),
only already-priced unblocked children. Do not infer this solely from source
comments or aggregate child counts.

GESDD query: MNTHR=INT(minmn*11.0/6.0) has TWO floating arithmetic operations;
add first DROUNDUP<=2 =>query bound4. Do not treat query as zero cost.
Execution: threshold2 +two DROUNDUP calls<=4 +SMLNUM/BIGNUM setup3 +ANRM NaNtest1
+input scale guards<=3 +output scale guards<=2 =>15 exclusive named units.
DLANGE adds3mn. At mostone input scale andone output scale add86+3mn+3k,
k=min(m,n). Hence GESDD exclusive+these helpers<=101+6mn+3k per execution,
with explicit generic scaling charge even when actual matrices need no scale.
No nested reductions or bidiagonal children included here.

Values-only DBDSDC N1 returns abs1. At N2..4, if lower bidiagonal it applies
N-1 DLARTG calls plus2 multiplications each (<=27(N-1)); then always calls
DLASDQ with UPLO=U,SQRE0 and no vectors. DLASDQ has no rotations in this domain,
calls DBDSQR, then performs N(N-1)/2 scalar comparisons for sorting. DBDSQR
calls Q1 and returns directly on observed INFO0; Q1 and descendants are already
priced. DBDSDC's final sort adds another N(N-1)/2 comparisons. Thus wrapper+
newrotationhelpers<=27(N-1)+N(N-1)<=93, also covering N1, per execution.
Conservative93 per GESDD allows lower rotations even for upper matrices.
Installed/source path equivalence and finite success premise require review.

Candidate remaining driver upper:4 per query +194+6mn+3k per GESDDexecution.
This deliberately upper-bounds missing native calls using exact saved GESDD
shape counts and does not infer new primitive census rows. Cold rank geometry
is charged once; large cold leggauss eigenvalue helper costs stay separate and
not magically included. No final fullcost claim from this candidate.
