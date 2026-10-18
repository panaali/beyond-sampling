# t353_score — T355: second scorer of T353 (frozen estimator_h2) vs T317f on DEV3

↑ [TASKS](../../../TASKS.md) · T353 (claude/t292-oracle-floor @29c15c68, budget_t317d/t351, README "T353") · T343 (../t339_check)

**Scoring only (no tuning). All numbers unverified.**

## Rule (fixed before running; DRI T355)
- **Sources:** hash-check the frozen sources against the freeze-commit prefixes. cont_f must equal the frozen T317f
  (327dbb4a…).
- **Run:** estimator_h2.est and T317f (cont_f.est) on all 100 DEV3 models (weights @e2322a6b, open truths @8134ab47).
- **Metrics:**
  - RMS (noise-corrected by the truths' SE^2), bias, error sd, corr;
  - C = FLOPs as executed / 5948 (mean and range);
  - F = MSE·max(C, 1000)/mean p(1-p).
- **Verdict:** whether MSE x C(T353) < MSE x C(T317f).
