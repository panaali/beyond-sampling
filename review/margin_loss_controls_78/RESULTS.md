# T259 numerical controls: independent saved review pending

The sole fixed run passed all 17 exact-rational cases. It made 36 linear-system attempts and returned 35 coefficient matrices; the remaining attempt was the preregistered singular-system rejection. All 36 progress events are retained. No fixture or tolerance changed and no run was repeated.

The rectangular and rank-deficient readout examples commute with the common ridge map. Shared intercept, common-mode shifts and fixed cell/pooled routing preserve the expected margins and first-index labels. All eight combinations of pair and sample normalization reproduce the same centered coefficients when the penalty is matched.

The separating examples also passed: unmatched regularization changes the query winner from 0 to 1; unequal output penalties break readout commutation; arbitrary singular nullspace choices change query predictions; clipping hidden predictions changes the winner from 1 to the first-index tie owner 0. Thus the controls preserve the theorem's boundaries rather than treating every margin-loss pipeline as equivalent.

## Evidence and accounting

- Frozen numerical source: 06c7cf14; documentation-only precision correction: 22b2f8e4.
- Independent source/resource review: 78ed0a7b, published 54a92781.
- Immutable execution gate: c3dd1878, successfully published 42239a27 before the separate launch.
- Raw run: a772c30a, published 3d873cc2. `output/results.json`, `output/progress.jsonl`, `run1.log` remain unchanged.
- Internal CPU: 0.787153895 seconds. External Bash time: 0.83 wall, 0.76 user, 0.07 system seconds. Peak RSS: 20,048 KiB. Actual caps: 16 soft/20 hard CPU seconds and 2 GiB address space, with 25 wall seconds after the review lock.
- Zero trained-data/model/checkpoint/forward/RNG/Gaussian/DEV2/HP1/HP2 calls. Exact solve counts are partial research accounting, not native-FLOP or inference-cost estimates.

These are finite numerical implementation controls for an independently reviewed exact theorem. They do not measure network agreement, establish a 99% ceiling, rule out weighted/top-rival/nonquadratic losses, or authorize a trained comparison. The independently coded saved arithmetic audit remains a separate prerequisite for the bounded handoff.
