# Installed 4x4 float64 determinant category envelope

The pinned NumPy binary's `det<double,double>` at0x5cb0 calls `scipy_dgetrf_64_` (0x5e97), tests info, counts pivot parity, then tests each diagonal sign, calls log (0x604f) and adds it to the accumulated logarithm. It calls exp (0x607c) and multiplies by the sign (0x6081). Singular-info branches bypass diagonal logs and still call exp(-infinity), multiplying by zero. Public NumPyv2.5.4 source1041–1070/1155–1186/1240–1280 agrees with these blocks. No full build-source equivalence is claimed.

Installed OpenBLAS exported GETRF at0xb0e20 checks dimensions; at0xb0f53 the threshold is m*n<=9999. Thus4x4 dispatches at0xb0f76 to `dgetrf_single`0x41d7c0. Its existing independently reviewed T150 small path enters `dgetf2_k`; runtime binary hashes match. T150's56 arithmetic envelope and T157's98 numeric pivot/guard envelope therefore cover this same LU work. The latter consists of40 absolute-value lanes and58 comparison/max lanes (including packed inactive lanes), excluding the four triangular beta comparisons specific to inverse solves. T157 final review must accompany use of this dependency.

| Category | Per determinant upper | Scope |
|---|---:|---|
| LU arithmetic |56|Reviewed small installed LU path; no inverse solves|
| LU absolute value |40|T157 IAMAX32 plus pivot-guard8|
| LU comparison/max |58|T157 IAMAX44 plus pivot8 plus scaling6|
| Reconstruction add/multiply |5|Four accumulated logs plus final sign multiply|
| Reconstruction sign negation |8|At most two per negative diagonal; binary XOR, counted as source negation|
| Reconstruction numeric comparison |4|One diagonal `<0` per successful pivot|
| log calls |4|Scalar libm calls, internals unpriced|
| exp calls |1|Scalar libm call, including singular branch|
| Numeric named subtotal |176|Upper on these categories only|

Separately the source pivot/info wrapper uses at most4 pivot inequalities,4 `change_sign` integer accumulations,1 remainder,1 parity truth test,1 info equality:11 named source operations. This yields187 per call if these integer source expressions receive unit cost. Loop counters, `i+1` indices, offsets, general dispatch/control, memory/allocation, BLAS copies and libm internals are not included. SIMD reduction may execute different integer lane work; the11 count is explicitly a source-expression budget, not instruction count or hardware envelope. It must not silently be combined with a hardware metric.

Transparent controls intercept GETRF/log/exp and prove bitwise output parity on18 fixed fixtures (19 matrices including batch). Two singular fixtures call0logs+1exp;17 successful matrices call4logs+1exp, including NaN/inf exceptional inputs. The normal/scaled finite cases also match independent24-term rational determinants within frozen tolerances. Overflow returns infinity and underflow zero as expected. Finite subnormal log-exp roundoff is retained; this audit is not a determinant accuracy certificate. No trained covariance or cone is evaluated.

Composition uses only the pinned parent census:35098 exact calls and28000 recipe calls. Numeric subsets are6,177,248/4,928,000 units; including11 pivot source expressions gives6,563,326/5,236,000. Add these to the parent visible categories only as a partial accounting total. Do not present these subset upper bounds as either full estimator upper bounds or full estimator lower bounds. Inverse cost, rank cost, leggauss and other excluded runtime categories remain separately owned/unpriced. The original visible parent mass operations exclude the entire det call, so there is no double counting with their multiplication by its returned value.
