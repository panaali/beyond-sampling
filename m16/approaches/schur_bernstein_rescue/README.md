# Stable Schur representation rescue (T162)

## Question
Does accurate coefficient preprocessing plus Bernstein numerator evaluation remove the specifically diagnosed T160 cancellation without per-node inverse calls?

## Method
Claim11e61fbc pushed first. Preserve T160 raw failures and sources. Exact Fraction arithmetic constructs all coefficients from the same stored binary64 C, then rounds each once. Write u=1−t. For complement pair i,j and remaining k,l, numerator is t*u²*Ckl + t²*u*[Ckl(Cii+Cjj)−Cik*Cil−Cjk*Cjl] + t³*P(C), where P(C) is the exact Schur cofactor numerator. Principal minors use the analogous u³, u²t, ut², t³ determinant expansion. No assumption that rounded C is exactly PSD or unit diagonal. No source fallback or covariance modification.

Freeze the same320 exact identities, three analytic masses and five near-singular covariance controls as T160. Require maximum conditional-correlation error<=1e-12 against exact equicorrelation at all120 nodes before trained work. Then, only after independent source/control review, permit the original256 evenly spaced seed0 archive indices floor(j*35097/255). Keep24/96 rules, source threshold/cutoff and initial covariance operations. Retain all raw correlation excursions and invalid variances; source clipping only after recording. Compare per-node and integrated differences versus the source, and compare saved positive reference without new integration. Any failures stop expansion; max integrated source deviation<=1e-10 is only a practical parity screen, not a roundoff certificate. No full corpus without new scope. T149 stays unchanged.

Alternatives: polynomial evaluation cancellation (Bernstein basis test), inaccurate coefficients (exact preprocessing test), covariance conditioning (same C isolates it), or excessive preprocessing cost (record rational setup separately). Exact coefficients may still produce signed-term cancellation for general C, so analytic success alone is insufficient.

## Status
Protocol frozen. Controls pending; trained panel requires independent gate.

## Results
Pending. Reproduce with project interpreter, one-thread limits and review lock for<=20CPU phases.

## Compute accounting
Count sixteen exact binary64-to-Fraction inputs, thirty computed/rounded coefficients, rational bit sizes, construction time and node evaluation separately. Fraction arithmetic is not priced as unit binary64 operations. Comparator inversions and exact controls are research costs, not hidden production-free work. No full portable-cost claim; one thread2GB and55CPU maximum primary phase.

## Claims
[inferred] Exact preprocessing removes coefficient construction rounding before the single float conversion. [speculative] Stable evaluation may make the inverse-free kernel practical; no competitiveness claim.

## Alternatives not ruled out
Guarded source fallback, interval/compensated preprocessing, signed cancellation control, direct2x2 Schur updates and positive integration.

## Failure modes
Rounded C indefiniteness, tiny principal minors, signed Bernstein numerator cancellation, Fraction setup expense, source arithmetic error, diagnostic-corpus selection and clipping concealing excursions.
