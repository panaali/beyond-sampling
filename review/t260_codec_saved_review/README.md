# T260 saved codec review

## Question
Do the eight retained-pilot frames faithfully encode their declared executable
syntax, selected coefficients and comparator dependencies at the reported length?

## Method
`reader.py` independently parses packed bytes and stdlib NPY records. `check.py`
reconstructs source ASTs and exact archive payloads, checks all 32 saved controls,
category lengths, dependency closure, overhead and progress. See `PROTOCOL.md`.
Alternative explanations are missing dependency/readout costs or lossy/misframed
roundtrip. Exact archive bindings and full tag consumption distinguish both.

## Status
Checker preparation only; source/resource gate and once-only saved audit pending.
No numerical result or coefficient payload has been decoded during preparation.

## Results
No saved audit executed. Sole command after published gate:
`bash review/t260_codec_saved_review/launch.sh`.

## Compute accounting
Fixed eight frames, at most240M bits, 23 archives/68 arrays,32 saved controls.
Reviewer cap95internal/105soft110hardCPU2GiB/120wall primarylock. Zero author,
predictor, fit, score, model, RNG or Gaussian calls. Reviewer work is not deployment
cost; inherited model preparation remains charged separately.

## Claims
[inferred] The checker can establish declared-code consistency, not minimality or
population agreement. [measured] No numerical audit result yet.

## Alternatives not ruled out
Shorter representations, changed precision, fresh validation, alternative
predictors and original historical models remain outside this fixed audit.

## Failure modes
Cap/metadata/array/schema mismatch stops and preserves the attempt. AST schema is
Python-version dependent. The declared code assumes its specified library/runtime
semantics; byte roundtrip alone does not prove numerical predictor equivalence.
