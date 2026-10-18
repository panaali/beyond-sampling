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

## Step 1 results (unverified; `python e0e1.py DEV3/weights.npz DEV3/truths.json ../t339_check/dev3.json` -> e0e1.json, e0e1_summary.json)

| quantity (DEV3) | mine, 100 models | T356 | mine, first 10 |
|---|---|---|---|
| E0 (true v4): bias / sd / RMS | -.139 / .412 / **.435** | -.129 / .414 / .432 | -.084 / .428 / .436 |
| E1 (node v4): bias / sd / RMS | +.167 / .423 / **.455** | +.173 / .431 / .462 | +.302 / .322 / .441 |
| T317f - E0: mean / sd / RMS | +.564 / .414 / **.700** | +.554 / .413 / .690 | +.778 / .341 / .849 |
| corr(T317f - E0, E0 - truth) | **-.003** | .00 | -.45 |
| quadrature check sqrt(sd^2 + sd^2) vs T317f sd | .584 vs .583 | .585 vs .586 | |

**Agreement criterion met on all four items (100 models).** [measured] T356's step-1 decomposition reproduces with
independent code and seeds.
- The 10-model subset is noisier: corr -.45, and integration RMS .85.
- The step-2 test uses the first 10, so the T317f line there is .849, not .690.

## Step 2 (pre-registered, before running): a non-walk integration scheme for closure c
**Choice and why.** T356's untried list:
- "Second anchor" and "second-order tilt" are bias-only by T356's own account, while the binding term is the sd of .41.
- Operator-norm screens and zonotopes cut cost only.
- J2 is costed at >= 21k.
- The remaining untried kind is a non-walk integration of the closure.

The cheapest representative is deterministic quadrature over the closure's own 4 coordinates, with the triple as a stratum:
- the closure verdict depends only on the triple (p1, p2, p3), which is uniform over the 720, and on (v1, v2, v3, v4);
- so E0 = mean over triples of ∫ verdict d(law of the top-4 order statistics).

**Scheme L (lattice).**
- **Lattice:** a rank-1 Korobov lattice in [0,1)^5 with N = 3203 points (prime), centred (+1/(2N)).
- **Generator a:** fixed model-free, the minimiser of the P2 criterion over a in [2, N/2].
- **Point mapping:**
  - coordinate 0 picks the triple, floor(720 u0);
  - v1 = Phi^-1(u1^(1/10)), v2 = Phi^-1(Phi(v1) u2^(1/9)), v3 = Phi^-1(Phi(v2) u3^(1/8)), v4 = Phi^-1(Phi(v3) u4^(1/7));
  - the closure-c input is built and the net is run once.
- **Estimate:** the mean verdict. It is deterministic and sampling-free: constructed inputs, no random draws.
- **Cost:** N forward passes, plus the mapping charged at 40 FLOPs per order statistic. Total ≈ 3.23k FPE, within 3.3k.

**Reference:** E0 by paired MC with 2e6 inputs per model (rng 3570 + index), giving ~.02 pp noise, on DEV3 dev3_s500..509.

**Overturning test (DRI / T356 §5):** RMS(L - E0) <= .15 pp and abs(mean(L - E0)) <= .1 pp, at C <= 3.3k FPE executed, on
the 10 models.

**Control:** plain MC with the same N (rng 3571 + index). Expected RMS sqrt(.0689/3203) ≈ .46 pp.

**Prediction (written now):** fail. A discontinuous verdict gives lattice rules little more than stratification, so I expect
RMS ≈ .3–.45 pp.
