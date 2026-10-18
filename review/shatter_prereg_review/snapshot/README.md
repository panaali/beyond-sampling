# shatter_test — T239: is the error part "shattered" while the correct part is simple? (diagnostic)

↑ [TASKS](../../../TASKS.md) · [width16_probe](../width16_probe/README.md) (T206 anatomy) · [random_control](../random_control/README.md) (T232)

**All claims unverified.** Diagnostic only; no estimator.

## Question (owner's intuition)
With only 16 neurons the network learns a structured approximation. The correct part is simple, and the error part is
"shattered": it behaves like a random network and creates the huge region counts.

## Pre-registration (written 2026-10-18, before any T239 number was computed)
**Data.** Exact final leaves of estT exact mode (chamber start, as T206) for zoo seed 0 at (8,3), (8,4), (16,3), (16,4).
These are re-run with leaf rays saved (T206's npz has masses but no rays). T192 has no per-leaf (16,5) data, so the T-growth
comparison is T = 3 -> 4 only. Prefix leaves (k = 4) of the target and the 12 T232 controls are re-run with rays and masses.
Per leaf: correct mass c = leaf mass - error mass e (exact).

**(1) Coverage by part.** Sort leaves by c (resp. e). N_c(q), N_e(q) = cells covering q = 90 / 99 / 99.9% of the total correct
(resp. error) mass. **Supports** if N_e(q) >= 3 N_c(q) at q = 99% for both H at T = 4, AND N_e grows faster than N_c from
T = 3 to 4 (ratio of growth factors >= 1.5). **Disproves** if N_e(q) <= N_c(q) at 99%, or N_e grows no faster than N_c.
Otherwise mixed.

**(2) Density by tie distance.** Each leaf's center u = normalized mean of its unit rays; tie distance
d = min(u_(1) - u_(2), u_(2) - u_(3)) / sqrt2 (angular distance to the nearest class-changing tie plane); blind = u_(2) <= 0.
Strata d < 0.02, 0.02-0.05, 0.05-0.1, 0.1-0.2, >= 0.2, plus blind. Report cells, mass and cells per unit mass per stratum.
**Supports** if cells per unit mass in d < 0.05 is >= 10x that in d >= 0.1 (non-blind) at T = 4, both H. **Disproves** if
< 3x. Otherwise mixed.

**(3) Controls at prefix k = 4** (no labels): the same density-by-tie-distance profile for the target prefix and the T232
random / scrambled models. Shattering "like a random network" predicts that the trained near-tie density approaches the random
models' density, while far from ties the trained density is lower. **Supports** if the trained far/near density ratio is
<= 1/3 of the random models' median far/near ratio. Otherwise not supported.

**Alternatives.** Near-tie density may be high for purely geometric reasons. The start partition (chamber halves) and the
x_i = x_j planes meet at ties, so cells there are thin regardless of training. (3)'s random controls separate that.
"Error cells are small" may also follow from error being a thin sliver along ties (low mass per cell) without any shattering.
(1) normalises per unit mass, but both readings predict large N_e.
