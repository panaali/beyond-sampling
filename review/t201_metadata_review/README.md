# T201 complementary source and metadata review

## Question
Do the thirteen frozen snapshots and cited helper origins support the limited
provenance inventory, without recovering historical fit/test numbers?

## Method
Assignment754e7160 preceded sourcebc264eed. Independent file SHA256 and Git
blob reconstruction bind all thirteen snapshots byte-for-byte to cf62f4ab.
AST inspection checks helper functions and module-level import side effects;
three cited source-origin file lists and the saved label-experiment JSON are
checked separately. No frozen module is imported. T55 separately owns the
blind-region/state/crossing theory; it is not duplicated here.

## Status
Complete after source gated5e60817 reached shared013b311c. One saved-only run
passed; results.json and run1.log are retained. No failure or retry occurred.

## Results
All13 SHA/blob/original-commit bindings pass. The state encoder is a helper;
separable.py contains design and supplied-mask least squares, without a driver
or train/test split. Features/gates/per-input linear-map helpers are distinct
from a reconstructed lookup fit. common.py loads a model at import;
estimate.py and arc_eval.py create400,000 reference inputs at import, which
this review avoided. The cited origin commits add only the listed helpers or
narrative. The saved JSON's replacement accuracy is.54133, not a lookup result.

Rounded prose totals4.68% in FINDINGS and3.75% in the shorter RESULTS list are
arithmetically consistent with the inventory;4.69−3.75=.94 percentage points.
These are contractions of printed decimals, not recovered empirical counts or
proof both reports used the same samples or mutually exclusive categories.

No coefficient table, fit split or raw cross-tab is recovered from these thirteen
snapshots. This is a local evidence-scope statement. I did not independently
repeat the author's broader all-ref/workspace search and assert no global absence.
Missing provenance is not proof of leakage or evidence against a new held-out fit.

## Compute accounting
One <=20CPU/2GB/one-thread review-lock metadata run; measured Python process
CPU.013540271. Read-only Git subprocess overhead is not included in that timer.
Zero author imports, model/noise/fit/truth/mass/reference calls.
Reproduce in fresh exclusive output paths with
`python review/t201_metadata_review/check_saved.py`; outputs refuse overwrite.

## Claims
[measured, source metadata] The snapshot bindings and limited helper/narrative
inventory pass. [inferred] These artifacts alone do not validate historical
98%/90–91% held-out claims. No new performance result or execution approval.

## Alternatives not ruled out
A historical driver may exist in another archive or uncommitted session.
A new frozen reconstruction can perform well; it cannot repair historical byte
or split provenance. T55's separate theory and any later experiment keep their
own assumptions and review gates.

## Failure modes
Treating a helper as a complete experiment; importing side-effecting sources;
turning rounded prose into raw counts; conflating a missing record with leakage;
using a metadata pass to authorize model fitting or a challenge/rung claim.
