# Candidate small BLAS envelopes (not yet accepted/contracted)

Scope: children of the three saved reducer shapes. Left applications have
GEMV transpose m=L-1<=23,n=C<=3. Right applications have GEMV no-transpose
m=C<=5,n=L-1<=3. All input strides are positive; GEMV output stride is1,
alpha1,beta0. Work is small and single-threaded. DAXPY lengths<=5 at these
applications; DGER m<=23,n<=3 or m<=5,n<=3. Source/control/memory exclusions
and historical fixed-binding premises remain. FMA counts as two scalar units.

DAXPY actual SKYLAKEX body is exactly one FMA per selected element for these
small positive-stride paths (scalar tail, four-scalar unroll, contiguous16
block with no inactive tail lanes). Fortran wrapper adds alpha-zero compare.
Proposed2N+1. GER kernel multiplies each y_j by alpha once, then directly calls
the DAXPY kernel (no extra wrapper comparison), optionally copying strided x.
Copy is memory-only in named units. Its wrapper adds alpha-zero compare.
Proposed2MN+N+1; includes every selected FMA lane.

GEMV wrapper performs beta-one and alpha-zero comparisons. It calls DSCAL kernel
for beta0 with KEEP_NAN=0 (zero fill), but conservatively allow output_length
multiplications plus two kernel comparisons. Wrapper allowance output_length+4.

GEMV T: m<2048,n<=3 invokes at most one block of B=4floor(m/4) rows; only4x2
and4x1 microkernels. Their multiply/add lane total is2Bn plus at most4n final
horizontal reduction lanes. Applying alpha and accumulating outputs adds2n.
Remaining r=m mod4 rows require r alpha-scalings and2rn multiply/add work.
Thus body<=2mn+6n+3, and including wrapper<=2mn+7n+7.
The stored microkernel disassembly matches these source loops; N-body4x4 and
add_y routines are not reached for n<=3 in this T path.

GEMV N: m<=5,n<=3. B=4floor(m/4) is0 or4. For contiguous x, installed4x2
microkernel uses5B arithmetic units (two products, sum, alpha product, y add),
not merely4B;4x1 uses2B+1. For strided x, n repeated4x1 calls cost2Bn+n.
A safe block allowance is3Bn+n. The r=m mod4 tail costs2rn+2r. Hence body
<=3mn+n+2m, wrapper included<=3mn+n+3m+4. Output stride1 avoids add_y;
copying strided input is memory-only. Public macros and actual inline code
must be checked against this path, and any additional executed inactive lanes
would need retention. No GEMV claim is accepted from algebra alone.

These are candidates pending independent source review and frozen small ABI/
branch controls. Controls will distinguish scalar one-FMA expectations from
separate mul/add rounding, preserve all disagreements, and will not establish
all-input compiled equivalence or full runtime cost.

Assuming these child envelopes pass, worst-case full-dimension child cost per application is left4LC+8C+10 and right5CL+2C+2L+5 for L>1; compare separately with known scalar-path DSCAL C+3. Do not add parent scan counts or previously priced DLARFG inside these expressions. Shrunk dimensions only reduce the polynomial envelopes; finite actual data need not be rerun.
