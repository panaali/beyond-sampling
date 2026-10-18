# Bounded source-exclusive eigenvalue iteration work

This is a conditional source-expression envelope, not a full installed-native price. Public3.12.0 and3.12.1 DSTERF differ only in line wrapping here; their source files remain separately pinned. Installed path capture confirms six DSTERF calls of sizes6,3,3,6,10,16 and successful final results. Binary loop correspondence and child numeric costs remain separate obligations.

For N>=2, the source sets NMAXIT=30N and checks the global JTOT before each shift. Each shift increments JTOT once. A visit to the QL label50 or QR label100 either performs a shift, deflates at least one eigenvalue, or terminates its block/the routine. Thus <=32N visits conservatively cover <=30N shifts, <=N total deflations, and <=N terminal/block visits. No convergence assumption is needed: exhaustion reports INFO and exits. The outer blocks are disjoint and number<=N. Search and recurrence loops each have length<=N-1.

Count each written real arithmetic, abs, comparison, sqrt or SIGN as one named expression; integer/index/control, memory and child routines are excluded. A multiply-and-add is two. This convention is explicitly distinct from actual duplicated SIMD lanes or compiler/libm internals.

| Exclusive source piece | Conservative expression upper |
|---|---:|
| Initial EPS/safe-range arithmetic and square roots |6|
| Initial block split scan |8N(N-1)|
| Squaring subdiagonal entries |N(N-1)|
| Block norm tests plus endpoint magnitude comparison |6N|
| QL/QR small-offdiagonal searches |160N(N-1)|
| Shift construction and final two assignments |360N|
| Inner scalar recurrence, including numeric C!=0 test |390N(N-1)|
| Two-eigenvalue square roots and final failure scan |2N|
| Total |559N(N-1)+368N+6|

The recurrence charges13 per step: R add1, optional E product1, two divisions2, GAMMA expression4, diagonal expression2, numeric branch1 and maximum branch arithmetic2. A shift plus its final E/D assignments costs12 outside the recurrence, excluding DLAPY2. The search predicate costs5. Initial split predicate costs8, including two sqrt and three abs. All guards involving loop indices remain in the omitted integer category.

Child envelopes retained separately:3 DLAMCH calls; <=N DLANST; <=3N DLASCL; <=30N DLAPY2; <=N DLAE2; <=1 DLASRT. A complete native envelope must price these, prove which scale branches are reached or bound them, and establish installed loop correspondence. Returned eigenvalue parity establishes call semantics, not those prices. Zero-reflector observations concern preceding SYTD2; they do not eliminate DSTERF iteration work.

For the frozen six orders the exclusive source bound sums to240,946. This deliberately loose source bound may be useful for cost headroom, but it is not yet added to the native numerical estimator subtotal and does not close the remaining SVD/eigen/helper/runtime gaps.
