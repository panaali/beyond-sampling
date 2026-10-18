# Selected native numeric replacement candidate

This is conditional on the frozen binary, saved layout and historical dispatch applicability. It excludes generic control/index/memory/casts and libm internals. The old hybrid ledger and all failed probes remain unchanged.

## Norm4:10 versus inherited8

Current gotoblas slot0x368 resolves to ddot_k_SKYLAKEX. Positive contiguous length4 takes dot_compute scalar tail: four sequential FMAs (8 arithmetic units), plus DOUBLE_dot adds the returned value to its zero accumulator (1). Scalar norm adds sqrt (1). There are16 norms per wide cone. Four orientation n@Rays[i] dot products are deliberately not replaced here: they require their separate matmul1D dispatch applicability, although a candidate9-unit path is plausible.

## Array sums: one addition per input under selected compiled reduction

Source pairwise n<8 includes addition to−0, but actual DOUBLE_pairwise_sum at0x5447a0 loads the first entry and performs n−1 scalar adds. The128 block is n−8 loop additions plus7 merge additions; recursion adds child totals+1. Thus every nonempty native pairwise call has n−1 additions, and DOUBLE_add adds its return to the initialized accumulator: total n. The source-only short-row upper5 remains separately recorded; native row4 is4.

Selected C-order1000x4 row reductions therefore replace3000 with4000; contiguous thin1000 replaces999 with1000; six contiguous wide128 sums replace762 with768. No cast/buffer conversion is required for aligned float64 allocated arrays. The contiguous1D fast reducer directly calls this loop. Row NpyIter reduced axis length4 uses the same adapter; alternate non-reduction traversal over1000 output entries would perform4*1000 adds with no vector tail because1000 is divisible by four. Actual historical traversal was not newly traced. If this applicability is not accepted, retain only the source-category upper and leave native row contraction unknown.

## Six exact Python floats: conservative61 versus inherited5

All asn values are exact built-in floats because math.atan2 returns float and its surrounding scalar arithmetic remains Python float. Initial int0+firstfloat is one float_add arithmetic and one numerical error-sentinel comparison after PyLong_AsDouble; integer conversion/control excluded.

Each of five installed compensation steps at0x18eda10..0x18eda5f executes:
-6 scalar arithmetic operations: t, both two-operation correction candidates, and c update;
-two packed ANDPD absolute-value masks, charged2 lanes each=4 abs units;
-one scalar CMPNLESD numeric comparison.
This is11 per step; both alternatives execute, unlike source seven-unit selection. Mask selection bitwise operations are control, separately excluded. At exit0x18edaae..0x18edafb, charge one scalar abs mask, two numeric classifications (magnitude versus infinity, magnitude versus zero), and at most one final add=4. Total2+5*11+4=61. This conservative bound charges unused high fabs lanes; it is not a hardware runtime count. Other type branches are outside the exact-six-float contract.

## Replacement only

Thin delta = (4000−3000)+(1000−999)=1001.
Wide delta =16*(10−8)+(768−762)+(61−5)=94.
All other matrix/native/source terms and190197 model-free constants stay unchanged. These bounded additions do not close the remaining orientation dot, elementwise ufunc, libm, geometry/cold and runtime categories or establish any sampler ratio.
