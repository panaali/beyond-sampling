# Retained-evidence preparation

## Question
Which existing C1/C2/D1 records can be checked without repeating fitting?

## Method
Under published scope1d48eda4, copy exact Git JSON/log bytes from a9b09a14; inventory JSON field types/list lengths only. Search filenames under /workspace/work and /workspace/clean-neurons using rg --files --hidden --no-ignore for phaseC2_eval.npz, phaseC_eval.npz, *hinge*coef*, *cell*coef*. No archive decode or score contraction.

## Status
Prepared; saved numerical audit not yet authorized. Parent notified of missing local named archives so existing owner artifacts can be recovered if retained elsewhere.

## Results
inventory.json binds four JSON files and three logs. phaseD.log is absent at the pinned Git path. Filename search returned no matches under the two specified roots; other roots, filenames and owner sandboxes remain unknown. Schema recursion records every dictionary field but only the first list element's schema plus length, not uniformity of every element.

C2 stores twelve-family validation/evaluation tables, ten-step selected families, single/group/cumulative metrics and sixteen evaluation-ranked cumulative rows. D stores five confusion shares, conditional summaries, gap/pattern bins, and rank-picked integer vectors. It does not store raw labels/gates or fitted models. Parsing these JSON objects is not independent score reproduction.

## Compute accounting
Zero author, array decode, model, fitting, RNG or scientific arithmetic calls. Git retrieval, JSON schema parsing, hashing and filename search only. The saved-only audit below remains a proposal.

## Claims
[measured, metadata only] inventory.json records the retained paths/hashes and bounded search outcome. [inferred] Integer compatibility and cross-file consistency may be recoverable from aggregate rates and known N; this would not independently verify the generating samples or predictions.

## Alternatives not ruled out
Raw artifacts may exist in owner sandboxes or differently named files. Missing local metadata does not invalidate reported scores. A new independent fit is a later alternative if recovery fails, with its own resource and provenance plan.

## Failure modes
Rounded rates may not uniquely identify integer supports, especially conditional denominators. JSON shape inventory is not full schema validation. Historical runtime/source/model identity cannot be supplied retrospectively by current hashes.
