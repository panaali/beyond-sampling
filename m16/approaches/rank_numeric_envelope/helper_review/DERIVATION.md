# Reflector helper named numeric envelopes

## DLAPY2 and DLAMCH
Installed DLAPY2 calls DISNAN twice, each tail reaching DLAISNAN with one
floating comparison. Non-NaN path uses two abs masks, one minimum and one
maximum, up to two floating comparisons (minimum zero, maximum above overflow
threshold), then divide, square, add one, sqrt and multiply. Named numeric upper:
2 NaN comparisons +2 abs +2 min/max +2 guards +5 arithmetic/elementary =13.
NaN/zero/infinite exits only reduce this bound. MIN/MAX count one numeric unit
each. A machine-constant call occurs on every branch. Installed DLAMCH uses
LSAME character tests and returns precompiled constants: zero floating arithmetic,
with character/integer tests and loads explicitly unpriced. This is not zero
runtime cost. No floating work is hidden in LSAME; this ABI helper compares and
normalizes character codes with integer instructions (inspect/pin before controls).

## DSCAL
Current gotoblas+0x390 dispatch is dscal_k_SKYLAKEX. Wrapper tests alpha==1
with one floating comparison even on trivial integer early exits. Saved positive
lengths are <=94, below the million-entry thread threshold. Wrapper passes
KEEP_NAN=1 as the third stack argument (three pushes:1,0,0). Kernel checks
alpha==0 first, then for the nonzero branch checks alpha==0 a second time;
at most two kernel numeric comparisons. Nonzero alpha and zero alpha both
multiply each selected entry once under KEEP_NAN=1. Contiguous full blocks use
8-lane AVX512 multiplies; strided full blocks call the pinned inc_8 helper with
four 2-lane multiplies; scalar tails cover exactly the remaining entries.
Thus at most N multiplies+3 numeric comparisons, N+3 total. Alpha==1 exits
at the wrapper and costs only its comparison. This envelope is current-dispatch
conditional; historical binary/binding remains a premise. No discarded SIMD
lanes are charged because full blocks contain exactly eight selected entries.

## Reflector composition
With the independently reviewed finite no-rescale premise, each N>1 reflector
has at most one DLAPY2, two direct DLAMCH calls, and one DSCAL of length N-1.
A zero norm skips these; using active-reflector counts is therefore an upper
bound, not an equality. DLAPY2's own DLAMCH is included as zero floating work
and unpriced character control. Generic rescaling can add helper calls and is
not covered by this conditional composition. All N<=1 reflectors return before
floating work. Existing exclusive10 includes the direct SAFMIN division and
all other parent arithmetic; do not charge them again. DNRM2 is separately priced.

This closes only named floating/numeric categories for reflector helper paths,
not memory, index/control, character tests, runtime, other SVD helpers, or full
rank cost. Elementary operations count one declared unit, not hardware cycles.
