# t356_recheck — T357: rule-5 re-check of T356 (strict-bar exhaustion checklist for the T317 paradigm)

↑ [TASKS](../../../TASKS.md) · T356 (claude/t292-oracle-floor, budget_t317d/exhaustion/README.md)

**All claims unverified. Different agent, own code. T356's floors_dev3.py was not read before step 1's numbers were pushed.**

## Step 1 (pre-registered, 2026-10-18, before running): reproduce E0 / E1 and the decomposition
**Closure c**, as defined in T356 / T324 / T304 (README text only).
- For an input x: p1, p2, p3 are the positions of the three largest values v1 > v2 > v3. The label is p2.
- v4 is the 4th largest value. It is placed at P4, the latest position not in {p1, p2, p3}.
- The other 6 free positions get low = mu(v4) = E[Z | Z < v4].
- E0 uses the true v4; E1 uses v4 = E[v4 | v3], with v4 the max of 7 iid N(0,1) | < v3 (K = 1 node), and low = mu(node).
- The verdict is argmax(net(x')) == p2.

**Run:** all 100 DEV3 models, 2e5 N(0, I) inputs per model (rng 357 + index, independent of T356's seeds), paired E0/E1.
Truth = the DEV3 1e8 open truths. T317f = my T343 DEV3 estimates (t339_check/dev3.json).

**Reported:**
- for E0 and E1 against truth: bias, sd and RMS;
- for T317f - E0: mean, sd and RMS, plus corr(T317f - E0, E0 - truth) and the quadrature check;
- the 10-model subset (dev3_s500..509) as well.

**Agreement criterion with T356:** RMS(E0) within ±.03 of .432, RMS(E1) within ±.03 of .462, RMS(T317f - E0) within ±.03
of .690, and abs(corr) <= .2. This allows for MC noise of about ±.02 per model.
