# Material correction: inherited coordinates are v=2x

After the sole author run6327896d, independent saved reviewer recover_910 found a coordinate-unit mismatch during checker preparation, before decoding the outputs. Recover_h4 independently confirmed the source facts: inherited local_certificate/THEORY.md defines v=2x; probability.py evaluates each boundary through cdf(F(v,2)); application.py uses boundaries_v directly.

Our original source gate e15a87b3 failed to propagate this scale into the new gap predicates. It is preserved unchanged as a historical review failure. Its clearance does not establish the intended physical-gap labels in the completed run.

With B measured in v, the implemented condition4*(v_i-v_j)>=G, G in{1,2,4}, is x_i-x_j>=G/8. Both inner and outer predicates therefore concern physical gaps{1/8,1/4,1/2}, while output keys and protocol claim{1/4,1/2,1}. The correct inequality for the intended physicalG/4 is2*(v_i-v_j)>=G, with the corresponding factor2 on all outer-interval endpoints. Alternatively one can explicitly work with rational x=v/2 endpoints. Changing only probability masses or relabeling old output as the intended grid would be incorrect.

The exact margin and mass provenance does not repair this mismatch. The zero-gap branch is unaffected. The positive-gap whole-box algebra is structurally valid for the implemented physical thresholds, but full saved-data validity still awaits independent review. The reported outside-cube tail is the inherited physical cube[-4,4]^4, not[-8,8]^4.

Preserve all source, gate, raw results and review chronology. No author calls, archive decoding, recalculation or rerun occurred in this correction. Any recovery must publish its scope and source gate first. A saved-only relabel audit could qualify the original observations; the intended gap1 row is absent and cannot be inferred from monotonicity as a measured row. New threshold evaluation, if desired, is a separate bounded recovery rather than a silent retry.
