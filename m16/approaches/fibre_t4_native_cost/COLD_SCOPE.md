# Cold setup retained outside the first-component ledger

The original loader imports fibre4, fibre4b, fibre and fiber3. In addition to
Python/NumPy imports, the following model-independent work actually runs:

- fibre4 builds24 face families. Six calls to `_basis` use full SVD on a1×4
  normal; each of24 faces uses two3×3 inverses in `_lune_tris`, hence48 inverses.
  Polygon normalization, small dense products and family bookkeeping also run.
- fibre4 calls leggauss at orders6,3,3,6,10,16: the original default rule,
  original correction rule and four adaptive cached rules. None of these six
  native eigensolver setups is currently included in the component ledger.
- fiber3 constructs its six T3 families on import, although T4 uses only its
  Ops class and imports rows_at through fibre. These constants are not advice
  from the model but are real implementation setup.
- fibre4 imports unused `_clip_raw` from m43, whose module reads the unrelated
  m43_weights.npz checkpoint into Wi0/Wh0/Wo0. The T4 numerical source never
  refers to `_clip_raw` or those weights. Removing the incidental import would
  need an owned, independently checked model-facing adapter; this audit does
  not silently remove it or charge it as free setup.

These are a read-only source census, not priced native envelopes. Repeated
review process imports must not be multiplied by48 when reporting one cold
estimator call; one actual cold setup must nevertheless be retained. Any later
removal of unused setup is a separate implementation change with output
parity/cost review. The first-component trace cannot close these categories.
