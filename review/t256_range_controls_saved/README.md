# T256 independent saved exact-range controls

## Question
Do the frozen synthetic affine-range results, witnesses and coordinate adapters agree with exact polytope geometry?

## Method
Assignment ba6ac722. `check.py` independently lists all active-constraint subsets and uses permutation determinants with Cramer's rule, distinct from the author's Gaussian-elimination reference. It checks the full feasible vertex set, extrema, reference system counts, every candidate completion/objective, selected witnesses, physical/v/saved-v adapters, fixed fixture identities, eight malformed-input rejections, partial expression counts and 42 progress events. The twelve fixed domains are written explicitly in the checker; no author modules are imported or called.

## Status
Checker source frozen before numerical output inspection; source/resource gate and immutable completed-input bindings are required before one saved run.

## Results
None yet. Exact affine ranges on touching or lower-dimensional domains do not imply positive Gaussian probability or label ownership.

## Compute accounting
One planned 19 soft / 20 hard CPU second, 2 GiB, 25 post-lock wall second review process. Determinant and exact-rational work is reviewer compute, not production estimator cost. No target/model/checkpoint, CDF, RNG, probability integration or author calls. Author expression counts are checked under their declared partial convention, not upgraded to native FLOPs.

## Claims
- [inferred] An independent vertex construction can falsify the breakpoint primitive and its coordinate conversion on the fixed controls.
- [inferred] Passing finite controls and a theorem does not establish trained certificate improvement or probability-mass accuracy.

## Alternatives not ruled out
Dependent ReLU remainders can dominate an improved affine range. A later partial-domain application still needs valid probability mass and old-certificate fallback; neither is tested here.

## Failure modes
Coordinate-unit mismatch, incomplete candidates, infeasible witnesses, wrong constraint signs or malformed-input acceptance can invalidate these controls. Every failed review attempt will be retained. No automatic retry or expanded fixture selection is authorized.
