# r1_walk (T304): add v4 to the walked R0 integral; can lenient F <= 10 be reached on retrains?

↑ [r0_walk (T302)](../r0_walk/README.md) · T298 (branch claude/t241-neuron-account, `r0_variants/`) · [TASKS](../../../TASKS.md)

**Scope (declared post hoc): lenient bar, DEV2 retrains.**

## Feasibility arithmetic (before any run)
- T302: integration ~.28 pp, total .67-.75 pp at 26-33k FPE, F ≈ 20.6.
- F = RMS^2 x C / .0725. At C ≈ 26k, F <= 10 needs total RMS <= ~.53 pp; at C ≈ 50k, <= ~.38 pp.
- T298's top-4 closure (floor .389 pre-registered) keeps v4 at its **true position**. That is 7 positions per triple
  (5,040 position tuples), so a walked integral costs ~7 x 2-3 (v4 Gauss nodes) x T302's ~26k ≈ 4e5 FPE. Then F
  ≈ .0048^2 x 4e5 / .0725 ≈ 120 even at RMS .48 pp. **Exact v4 position is infeasible at this cost.**
- The cheap form needs a **declared position** for v4 per triple. That is a new closure whose oracle floor is not
  known.

## Pre-registration (2026-10-18, before running)
**Stage A (diagnostic oracle floors, paired MC on all 80 DEV2 retrains; 200k skeletons per model, own code).**
Closures:
- a: R0;
- b: top-4 at its true position (T298 b);
- c: top-4 with v4 at the **latest** free position;
- d: top-4 with v4 at the **earliest** free position.

In each, the remaining 6 lows sit at mu(v4) = -phi(v4)/Phi(v4). Floor = RMS over models of (closure accuracy -
truth). b and a are checks against T298 (.389, .515).

**Stop rule:** if min(c, d) >= .48 pp RMS, stop. A declared-position closure cannot then give F <= 10 at any cost
>= 26k, since total >= floor.

**Stage B (only if Stage A passes):** walked integral (T302 structure) with the declared v4 position and v4's value
by a 3-node Gauss rule on its exact conditional law given v3 (numerical integration of a 1-D conditional law, not
sampling). Plus the DRI's cost cuts. **Go:** lenient point F <= 10 on all 80 DEV2 at own cost. Every step deductive
or declared.

## Stage A result (`python floors.py 200000`, `results/floors.json`; 80 DEV2, paired, 200k skeletons each)

| closure | RMS floor (pp) | mean (pp) |
|---|---|---|
| a: R0 (T298: .515) | .510 | +.215 |
| b: top-4, true position (T298: .389) | .390 | +.284 |
| **c: top-4, v4 at the latest free position** | **.471** | -.156 |
| d: top-4, v4 at the earliest free position | .557 | +.365 |

min(c, d) = .471 < .48, so **Stage A passes (barely)**. The checks a and b reproduce T298.
**Hard bound:** with floor .471, F <= 10 needs C <= 32.7k FPE even with zero integration error.

## Stage B pre-registration (2026-10-18, after Stage A, before building Stage B)
**Cost diagnosis** (T302 at tmax .5, one model): the walk costs are nonblind g23 3.9k, nonblind g12 3.5k, blind g23
2.3k and **blind g12 14.7k**. Near v2 ≈ 0 the inputs straddle zero, and the gates flip constantly.

**Estimator R1W.** The T302 walk structure, with closure c:
- v4 at the latest free position of the triple;
- v4's value by a K-node Gauss rule for its exact conditional law given v3 (density 7 phi(v) Phi(v)^6 / Phi(v3)^7 on
  v < v3; nodes from the moments of that law; numerical integration of a 1-D conditional law, not sampling);
- along the g23 line, v4 moves with v3 (v4 = v3 - delta_node; declared), and the 6 lows are mu(v4) linearised
  (declared);
- **in the blind region the g12 walk is dropped** (g12 assumed correct; declared);
- walk length tmax = .5 (T302's best).

**Variants (fixed now):** **V1** K = 1 (primary); V2 K = 2; V3 K = 2 with the blind g12 walk kept.
**Go:** V1 lenient point F <= 10 on all 80 DEV2 at its own cost. V2 and V3 are reported. If V1 fails and V2 or V3
passes, that is reported as a secondary pass (a 3-way selection on one panel).

## Stage B results (`python f80.py`, `results/f80.json`; all 80 DEV2 retrains, 1e8 truths, own charged cost)
Gauss rule for v4 | v3* = .665 (model-free check: weights sum to 1): K = 1 node .4025; K = 2 nodes .0065 / .5241
(weights .235 / .765).

| variant | C (FPE) | bias (pp) | RMS (pp) | point F [95% CI] |
|---|---|---|---|---|
| **V1** (K = 1, blind g12 dropped; primary) | **13,473** | +0.445 | 0.795 | **11.7 [8.5, 15.4]** |
| V2 (K = 2) | 26,776 | +0.288 | 0.737 | 20.1 [14.1, 26.7] |
| V3 (K = 2, blind g12 kept) | 52,108 | +0.219 | 0.713 | 36.6 [25.5, 48.5] |

- **FAIL (narrow).** The primary V1 misses F <= 10. At its cost it needed RMS <= 0.73 pp and reached 0.795.
- V1 nearly halves T302's best F (20.6 at 26k) at half the cost, with every step deductive or declared.
- **More v4 nodes or more walks lower the error only slightly while doubling the cost**, so F rises.
- **Error budget [inferred]:** total .795 against closure c's floor .471 implies an integration component of about
  .64 pp (if independent). That is larger than T302's .28, from the extra declared approximations: the v4 node at
  its conditional mean, v4 moving with v3, the blind g12 walk dropped, and one transverse point per region. The bias
  is +.45 pp, while closure c's floor mean is -.16 pp, so the integration error is systematically positive.
- **What would pass [inferred, not tested]:** at ~13.5k FPE, an integration error of <= ~.57 pp (about 10% better
  than now). The candidates:
  - a short blind g12 walk;
  - a second transverse point instead of a second v4 node.

  Any of these would be a new pre-registered variant, not a re-tuning of this panel.

## Status
Done (2026-10-18). Stage A passed (floor .471); Stage B **failed narrowly** (best F 11.7 at 13.5k FPE). Nothing was
sent to HP2.

## Compute accounting
Stage A: 80 models x 4 closures x 2e5 forwards ≈ 6.4e7 forwards (diagnostic). Stage B: 3 variants x 80 models, ~25
CPU-min in all.

## Claims (unverified)
- [measured] The declared-position top-4 closure (v4 at the latest free position) has an oracle floor of .471 pp RMS
  on DEV2. True-position top-4 has .390 and R0 has .510; both reproduce T298.
- [measured] R1W (walked integral, all classes, deductive + declared): F 11.7 [8.5, 15.4] at 13.5k FPE, bias
  +.45 pp, RMS .80 pp. Variants with more nodes or walks have higher F.
- [inferred] The deductive R0-family route on retrains is within ~15% (in RMS) of the lenient line at ~13k FPE. Its
  limit is now the integration approximations, no longer cost or cancellation.

---

# T306: the last ~10% (pre-registered 2026-10-18, before any T306 run)
**Refinements of R1W V1 (K = 1; everything else unchanged), exactly these:**
- **A:** a cost-capped blind-region g12 walk. Walk length tmax_b12 = 0.10, at most 4 cells per line. Beyond the walked
  range, g12 is assumed correct (declared, as for the other lines).
- **B:** two transverse points per region. Each region's v2 range is split at its model-free median (nonblind:
  v2 in (0, m_nb] and (m_nb, inf); blind: (-inf, m_b] and (m_b, 0]). Each half gets its own conditional-mean
  transverse point (v2, g12, g23, and v4's K = 1 node given that v3), its own walks, and its own Gauss-Legendre mass
  integral over its v2 half.
- **A+B:** both.

**Selection rule (fixed now):** the lowest point F on DEV2 (all 80, 1e8 truths, own charged cost) among V1, A, B
and A+B.
- The winner is frozen as `est(model, budget) -> (estimate, cost_flops)` and sent to the T281 scorer for HP2.
- **V1 itself is also sent**, so HP2 shows whether the refinement helped.
- **The claim is HP2's score alone.** DEV2 is development data and is reported as such.

## Forking-paths note (every variant tried on DEV2 for the R0 family, by me)
- **T297:** 7 threshold ladders (n23 x Newton x blind), plus 4 cheap variants (tb_cheap); 10 DEV2 models.
- **T300:** stage 0, 4 frozen T297 designs on all 80 (the 1,752 one sent to HP2); stage 1, 3 sharing rules on 10.
- **T302:** walk length tmax in {.25, .5, 1, 2, 6} on 10 models, and {.05, .5, 1} on all 80.
- **T304:** 4 closures (floors) on 80; R1W V1 / V2 / V3 on 80.
- **T306:** V1 / A / B / A+B on 80 (this section).

In total about 30 variants have seen DEV2. HP2 is the only clean test.

## T306 results (`python t306.py`, `results/t306.json`; all 80 DEV2 retrains; **development data**)

| variant | C (FPE) | bias (pp) | RMS (pp) | point F [95% CI] |
|---|---|---|---|---|
| **V1** | 13,713 | +0.445 | 0.795 | **11.95 [8.68, 15.69]** |
| A (capped blind g12 walk) | 16,118 | +0.428 | 0.785 | 13.71 [9.97, 18.0] |
| B (two transverse points per region) | 28,350 | +0.780 | 0.983 | 37.8 [29.5, 47.3] |
| A+B | 33,797 | +0.767 | 0.973 | 44.1 [34.4, 55.4] |

(V1's FPE here is 1.7% above T304's 13,473, because the Gauss-Legendre work is charged per region in r1w2. The
estimates are identical.)

- **Pre-registered selection: V1** (lowest point F).
- A lowers the error by 0.01 pp at +2.4k FPE.
- **B makes it worse.** Splitting each region at its median v2 doubles the walks and *raises* the bias
  (+.45 -> +.78 pp). The one-point closure of the boundary per region was not the main error source, and the
  half-region transverse points are worse anchors for the outer v2 tails.
- The winner is V1 itself, so the "winner" and "V1" HP2 submissions coincide. `estimator.py` (frozen) was sent to
  the T281 scorer. **The claim is HP2's score alone.**
