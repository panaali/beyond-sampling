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
