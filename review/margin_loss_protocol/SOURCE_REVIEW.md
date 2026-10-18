# Pinned source contract

All five text files are exact Git bytes from a9b09a14; manifest.json records paths, SHA256, sizes and Git blob IDs. No source was imported. This is a scoped source reading, not a reproduction of author result tables.

- encode.py defines CellLinear(lam=1e-6). Each cell uses Lambda=lam*(trace(X^T X)/d+1e-12) I. Pooled fallback uses Lambda=1e-9*trace(G)/d I. Both depend on the design, not the target output. All columns receive the same penalty; declared bases have no intercept. Pooled fallback can be singular on a zero design; the theorem is conditional on successful solves or an explicitly shared linear generalized inverse, not a guarantee of source totality.
- Query unseen-cell routing depends on cell IDs alone. Per-cell and pooled designs are common across output columns. This is exactly the required output-linear structure.
- HingeBank fits clean-training comparator preactivations using FAMILIES[5], then uses ReLU of predicted comparator formulas. It is training-target-derived advice. Evaluation features are input-computable from the retained bank; they are not held-out true gate values. The bank must be frozen identically for the algebraic comparison.
- phaseC_diag.py forms training htr @ Wo.T then fits logits, and separately fits htr followed by the same readout. The query hidden predictions are not clipped. Hinge arrays are cast to float32 before both designs. Those shared arrays do not break exact-real commutation, but target formation, accumulation, solve and readout rounding can.
- phaseC2.py selects per-neuron/time families by validation R2, stores evaluation predictions as float32 and uses ReLU replacements. These are not a shared linear fit across all outputs; the theorem must not be transferred to C2 as an equality claim.
- No frozen result value is newly verified here. README performance statements remain author evidence subject to T258 review. Finite fit results do not establish a universal feature-set information ceiling. Prior same-summary counterexamples establish pointwise dependence, not a probability floor ruling out 99% agreement.

Internal repository evidence only; no new external ARC/AlgZoo source was consulted. The T258 source/provenance review and T260 description-length inventory remain separate ownership scopes.
