# Frozen stored-cofactor application

Renewed TASKS-only claim d46739b1 precedes implementation. Primitive controls independently reviewed cb751402. This phase contains exactly64 stored3x3 nonprincipal cofactors, with no native determinant, mass, model or quadrature calls.

The immutable latest T179 panel archive contains139 Rt arrays. Freeze its first eight numeric labels825,1101,1238,1513,1789,2064,2615,2752; four node indices0,42,85,127; and source pair positions0,1. Hashes, source and capture code are in manifest.json. Selection uses stored identifiers and source pair order, not newly observed accuracy.

For source pair(i,j), remaining indices are(k,l). The source creates ck by deleting k and cl by deleting l, computes ckl=(-1)^(k+l)*det(Rt[:,ck][:,:,cl]), then calls arctan2(-ckl,sqrt(dR*om)). Observer.arctan2 copies its two arguments before calling NumPy. Thus saved argument0 is exactly the floating negative signed cofactor; there is no intervening scale. Recover signed source point by negating that value. Recover unsigned determinant only by the explicit sign(-1)^(k+l). For pairs0/1 the deleted indices are(2,3)/(1,3), with signs minus/plus. These are nonprincipal cofactors; no PSD or nonnegative determinant assumption.

Apply the frozen primitive to the exact stored minor. Map its interval to the signed cofactor by reversing and negating endpoints when the sign is negative. Save every entry in hexadecimal, exact unsigned/signed determinant, source and candidate point errors, interval containment/width/zero status and primitive category counts. Signed wrapper negations are additional recorded work, not included invisibly in the primitive. An exact six-permutation reference is independent of the primitive's Laplace tree; external reviewer uses a separate exact method.

Two alternatives remain live: the source may already be more accurate than the simpler point formula; or most useful intervals may be wide/cross zero despite passing containment. Preserve both outcomes and all source/candidate ties. No mass-level accuracy, native speed, full-corpus or cheap300-unit claim. Outward intervals certify stored minors only, excluding rounding in the upstream Rt construction and every other integrand term.

Source gate before execution; independent saved audit before another phase. No saved data are modified and no original mass values are recalculated.

The frozen64 application completed after the independent source/sign-mapping gate. All64 used the fast enclosure, contained the exact signed determinant, and excluded zero; no exact fallback. Candidate point error is smaller than the saved source point in34 cases, larger in27, and tied in3. Maximum relative candidate point error is7.37902440685544e-11; the source maximum is1.3418550782687846e-10 on a different row. No universal point-accuracy ranking follows.

Maximum relative enclosure width is6.047830053966127e-9. This certifies the stored minor only and does not turn the error into a Gaussian mass bound. All64 have the dense primitive inventory31 arithmetic/9abs/28comparisons, plus9finite/2nextafter/9conversions. The signed wrapper uses96 additional negations across the32 negative-sign cases. No native cost ratio is formed.

Saved results.json contains all exact errors, hashes, provenance, signs and category counts. run1.log preserves the first execution. Recorded audit/reference CPU .013559756 seconds excludes imports and final serialization; zero native determinant/mass/model/MC calls. Independent saved audit pending. Reproduce `run.py` under the review lock in an isolated output copy; the script refuses to overwrite the saved result.
