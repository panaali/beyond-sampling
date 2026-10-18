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

## Method
`leaves.py` dumps exact leaves (error mass, leaf mass, center): `final H T` runs estT exact mode on zoo(H,T,0) (chamber
start, as T206), and `prefix MODEL 4` runs width16_probe/prefix.py on the T232 models (`random_control/rc.py`). `analyze.py`
-> summary.json. Leaf masses sum to 1 within 1.1e-10 everywhere.

## Results (unverified)
**(1) Coverage by part, final leaves.** Cells covering q of the correct / error mass:

| (H,T) | leaves | error mass | q = 90% | 99% | 99.9% | (2) near/far density |
|---|---|---|---|---|---|---|
| (8,3) | 452 | 6.55e-04 | 78 / 28 | 159 / 62 | 243 / 90 | 14.7 |
| (8,4) | 8,455 | 2.38e-02 | 382 / 495 | 1,315 / 1,542 | 2,494 / 2,789 | 27.1 |
| (16,3) | 1,162 | 6.10e-04 | 122 / 27 | 286 / 89 | 461 / 185 | 19.1 |
| (16,4) | 51,552 | 2.13e-03 | 1,346 / 954 | 5,262 / 3,944 | 11,209 / 8,084 | 26.4 |

At T = 4, 99%: N_e / N_c = 1.17 (H = 8) and 0.75 (H = 16). Growth T = 3 -> 4 at 99%:
correct x8.3 / error x24.9 (H = 8), correct x18.4 / error x44.3 (H = 16): the error part grows
3.0x and 2.4x faster. Per unit *absolute* mass the error part is of course far denser (error mass ~2e-3 at (16,4)), but
that is the small-mass effect the pre-registration normalised away.

**(2) Density by tie distance, final leaves:** cells per unit mass near ties (d < 0.05) / far (d >= 0.1, non-blind) =
14.7 / 19.1 at T = 3 and **27.1 / 26.4** at T = 4 (H = 8 / 16): >= 10x, and rising with T.

**(3) Prefix k = 4, cells per unit mass by tie-distance stratum** (no labels):

| model | leaves | d < .02 | .02-.05 | .05-.1 | .1-.2 | >= .2 | blind | near/far |
|---|---|---|---|---|---|---|---|---|
| target | 24,407 | 358,060 | 28,216 | 8,592 | 4,087 | 2,872 | 26,837 | 44.1 |
| frob0 | 29,009 | 45,123 | 27,525 | 26,592 | 20,567 | 18,909 | 40,644 | 1.7 |
| frob1 | 28,244 | 31,942 | 43,296 | 32,408 | 30,659 | 24,736 | 22,329 | 1.3 |
| frob2 | 27,932 | 47,430 | 33,912 | 29,541 | 19,672 | 15,461 | 34,685 | 2.2 |
| frob3 | 26,932 | 40,489 | 38,884 | 31,862 | 27,530 | 25,356 | 18,538 | 1.5 |
| frob4 | 27,623 | 69,526 | 24,587 | 25,409 | 24,088 | 23,514 | 26,026 | 1.7 |
| row0 | 30,157 | 135,317 | 22,654 | 21,279 | 14,232 | 12,145 | 36,487 | 4.9 |
| row1 | 22,532 | 161,841 | 21,442 | 13,193 | 8,525 | 14,011 | 19,812 | 6.6 |
| row2 | 27,247 | 90,178 | 22,144 | 15,823 | 11,015 | 10,939 | 39,048 | 4.6 |
| row3 | 25,773 | 169,632 | 24,472 | 12,864 | 13,312 | 18,646 | 23,763 | 4.7 |
| row4 | 30,418 | 97,118 | 22,510 | 18,080 | 13,301 | 13,111 | 37,820 | 4.3 |
| perm0 | 25,376 | 161,901 | 30,994 | 16,206 | 16,695 | 13,635 | 16,970 | 5.3 |
| flip0 | 23,587 | 130,262 | 12,191 | 12,879 | 9,522 | 10,275 | 24,987 | 5.5 |

Trained near/far 44.1; random median 3.3 (range 1.3-6.6); perm 5.3, flip 5.5. Trained far/near ratio = 0.07 of
the random median (rule: <= 1/3).

## Verdict (pre-registered rules)
- **(1) not supported.** At H = 16, N_e < N_c at 99% (0.75x), which meets a disprove condition; H = 8 is mixed (1.17x < 3x).
  But the error part's coverage grows 2.4-3x faster than the correct part's from T = 3 to 4. If that continues, it
  dominates at larger T [inferred, 2 points].
- **(2) supported.** Cells per unit mass are 26-27x higher near ties than far from them at T = 4.
- **(3) supported, but not "like random".** The trained prefix concentrates its regions at ties: 44x near/far vs
  1.3-6.6x for random and scrambled weights. Near ties, the trained net is 2.1-11x MORE fragmented than the 10 random draws (358k
  vs 32-170k cells per unit mass in d < .02); far from ties (d >= .2) it is 3.8-8.8x coarser (2.9k vs 11-25k).
- **Overall:** the owner's picture is right about *where* the complexity is (at the tie walls, rising with T, and caused by
  training), and wrong that it is "random-like": it is more fragmented than random, and the error mass is not more spread
  out than the correct mass at T = 4. With T232 (same total count as random), training *redistributes* regions from the bulk
  to the tie walls [inferred].

## Alternatives not ruled out
- Geometry: the chamber-half start and the order planes meet at ties, so final cells are thinner there anyway. The prefix
  test (3) has the same start for all models, and only the trained one concentrates, so this does not explain (3).
- Center-based tie distance is one choice; min over rays (any contact with a tie) was not used.
- T = 3, 4 only (no (16,5) per-leaf data); (8,5)/(16,5) would test the growth claim.

## Compute accounting
Final leaves (8,3)-(16,4) ~8 CPU-min; 13 prefix models at k = 4 ~25 CPU-min; analysis seconds. ~35 CPU-min total.

Per-leaf dumps (16 MB) live in data/t239/ (git-ignored; `out` is a local symlink). Regenerate with `./drive.sh` (~35 CPU-min).
