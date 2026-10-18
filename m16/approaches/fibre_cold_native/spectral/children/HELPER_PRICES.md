# Fixed observed spectral helper numerical categories

Candidate source/binary envelopes pending independent review; no mass/model work. All children refer to the observed12call pass; this is not a hypothetical fullSVD price. Parent and child categories remain separated.

## New installed scalar helpers

**DLAE2 <=28 numeric units.** The installed disassembly is acyclic and makes no child calls. Initial three add/sub operations, four two-lane ANDPD abs masks, and one COMISD cost12. The RT branch adds at most two COMISD plus divide,multiply,add,sqrt,multiply:7. The sign-of-sum branch adds at most two COMISD, two RT1 operations and five RT2 operations:9. Thus12+7+9=28. The zero-sum branch has only one multiply and a two-lane XORPD sign mask and is smaller. MOVs, zeroing PXOR, integer branches and memory are excluded. Worst-branch bounds need not be jointly attainable to be valid. See118bc33..118bd31. Six actualcalls contribute<=168.

**DLASRT <=N(N-1)/2 numeric comparisons for N<=16.** Source SELECT=20 and installed11a9c10 compares block length with20. The fixed six orders therefore take insertion sort, never quicksort. The increasing-order inner comparison at11a9d9f is one scalar COMISD; each outer I has at most I-1 comparisons. Swaps/copies and index comparisons are excluded. Sum for6,3,3,6,10,16 is201.

**DLANST('M') <=8N-6 numeric units.** Installed11947a5 charges two abs-mask lanes initially. Each of the remaining N-1 pairs has two two-lane abs masks, two scalar COMISD and at most two DISNAN checks:8 per pair. DISNAN→DLAISNAN at11af416 performs one UCOMISD per call; all its other work is integer/control. Thus2+8(N-1). The source’s other norm branches are unreachable for the captured 'M' argument. Six calls total316. All original D vectors are zero and E finite positive, but this envelope does not assume NaN checks are skipped.

## Reviewed inherited helpers

Owner56 confirms the pinned installed positive-stride NRM2 envelope: N1 costs oneabs; N>1 costs2N+4, including allzero inputs (norm review3365a28f, independentd90477c1). The38 actual calls consist of six N3 fullSVD calls and32 zero-vector eigenreduction calls with N1..14; sum484.

DSCAL <=N+3 for positive strides applies to12 observed N3 calls, stride1 or4:72 (helper reviewefef5c7d/62428988, independent0cfdfa39). DLAPY2 <=13 applies to91 observed calls:1183. These exclude integer/control/memory and retain the inherited installed-dispatch premises; exact files are pinned separately before composition.

The six helper groups therefore total2424 numeric units. The DSTERF exclusive body, fullSVD parent/BLAS bodies, Legendre utility/reduction internals and remaining runtime categories are not included or declared free. Generic DLARFG parent113 may be a separate reviewed upper, but it is not automatically bundled into2424. No full-cost or matched-budget claim follows.
