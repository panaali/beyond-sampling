# Stable Schur representation rescue (T162)

## Question
Does accurate coefficient preprocessing plus Bernstein numerator evaluation remove the specifically diagnosed T160 cancellation without per-node inverse calls?

## Method
Claim11e61fbc pushed first. Preserve T160 raw failures and sources. Exact Fraction arithmetic constructs all coefficients from the same stored binary64 C, then rounds each once. Write u=1−t. For complement pair i,j and remaining k,l, numerator is t*u²*Ckl + t²*u*[Ckl(Cii+Cjj)−Cik*Cil−Cjk*Cjl] + t³*P(C), where P(C) is the exact Schur cofactor numerator. Principal minors use the analogous u³, u²t, ut², t³ determinant expansion. No assumption that rounded C is exactly PSD or unit diagonal. No source fallback or covariance modification.

Freeze the same320 exact identities, three analytic masses and five near-singular covariance controls as T160. Require maximum conditional-correlation error<=1e-12 against exact equicorrelation at all120 nodes before trained work. Then, only after independent source/control review, permit the original256 evenly spaced seed0 archive indices floor(j*35097/255). Keep24/96 rules, source threshold/cutoff and initial covariance operations. Retain all raw correlation excursions and invalid variances; source clipping only after recording. Compare per-node and integrated differences versus the source, and compare saved positive reference without new integration. Any failures stop expansion; max integrated source deviation<=1e-10 is only a practical parity screen, not a roundoff certificate. No full corpus without new scope. T149 stays unchanged.

Alternatives: polynomial evaluation cancellation (Bernstein basis test), inaccurate coefficients (exact preprocessing test), covariance conditioning (same C isolates it), or excessive preprocessing cost (record rational setup separately). Exact coefficients may still produce signed-term cancellation for general C, so analytic success alone is insufficient.

## Status
Bounded controls and fixed256 panel complete after independent T55 source/control gate. Final saved-array review requested; no full-corpus expansion.

## Results
All320 exact identities, three analytic masses and600 near-singular node comparisons pass; maximum exact conditional-correlation error is1.11e-16. The original catastrophic T160 outputs remain preserved. The independently approved256 panel has no failures or raw-correlation excursions. Maximum integrated deviation from the unchanged doubled source is9.783840404509192e-16; signed deviation−1.7822657412402965e-15 and L1 deviation5.573612318204857e-15. Three negative masses remain in both methods. Their common quadrature error against saved positive values is effectively unchanged: sourceL1 3.406582186494579e-7 versus candidate3.4065821776989665e-7. This is numerical parity on this finite panel, not universal stability or a mass-accuracy repair.

Reproduce `controls.py` then `panel.py` with project interpreter, one-thread limits and review lock for<=20CPU phases. Source comparison values are checked bitwise against the independently reviewed T140 archives. All256 indices are fixed in advance; no new integration references were generated.

## Compute accounting
Count sixteen exact binary64-to-Fraction inputs, thirty computed/rounded coefficients, rational bit sizes, construction time and node evaluation separately. Fraction arithmetic is not priced as unit binary64 operations. Comparator inversions and exact controls are research costs, not hidden production-free work. Controls recorded .101542112CPU and panel1.625469041CPU including comparisons/diagnostics. Panel candidate calls total .561355317CPU versus source .626234844CPU; candidate exact coefficient setup accounts .069583889CPU. This single ordered run is not a benchmark speedup claim. Imports, archive loading, diagnostics and serialization account for the remaining recorded time. Maximum coefficient bit sizes and every setup time are saved per row. No full portable-cost claim; one thread2GB and55CPU maximum primary phase.

## Claims
[inferred] Exact preprocessing removes coefficient construction rounding before the single float conversion. [measured] The stable representation removes the diagnosed synthetic failure and preserves the fixed trained-panel values within9.79e-16. [speculative] A guarded kernel could remove repeated inverse costs; rational setup and full source-expression accounting remain unresolved, so no competitiveness claim.

## Alternatives not ruled out
Guarded source fallback, interval/compensated preprocessing, signed cancellation control, direct2x2 Schur updates and positive integration.

## Failure modes
Rounded C indefiniteness, tiny principal minors, signed Bernstein numerator cancellation, Fraction setup expense, source arithmetic error, diagnostic-corpus selection and clipping concealing excursions.
