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

## Results (unverified; `dev3.json`, `dev3_summary.json`)
All 5 source hashes match the freeze commit 29c15c68; cont_f = frozen T317f. The T317f estimates are identical to T343's
independent DEV3 run (100/100).

| arm (DEV3, 100) | RMS pp | bias pp | error sd pp | corr | C (FPE, range) | F | MSE x C (x1e4) |
|---|---|---|---|---|---|---|---|
| T317f | .721 | +.425 | .583 | .9775 | 3,307 [2,667, 4,216] | **2.49** | **1,720** |
| T353 (estimator_h2) | **.673** | +.340 | .581 | .9784 | 5,056 [4,068, 6,924] | 3.32 | 2,288 |

**Verdict: MSE x C(T353) = 2,288 > T317f's 1,720, so T353 does not beat T317f on DEV3.**
- RMS improves .721 -> .673, all from the bias: +.425 -> +.340. The error sd is unchanged (.583 -> .581).
- C as executed rises 53%.
- On DEV2 the bias fell to ~+.04. On DEV3 it falls only .085 pp.
