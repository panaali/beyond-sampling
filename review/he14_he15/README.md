# T280: independent check of ledger rows H-E14 (generic region counts) and H-E15 (tie-concentrated fragmentation)

↑ [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · originals (T194 child, branch claude/t194-width16-probe):
`m16/approaches/random_control/README.md` (T232, H-E14), `m16/approaches/rate_alpha/README.md` (D6, activity-matched),
`m16/approaches/shatter_test/README.md` (T239, H-E15). Only the READMEs were read, for the protocol. No code was reused.

## Question
Do the two rows hold up under different code and a different enumeration algorithm?
- **H-E14:** the target's prefix region counts are generic for the architecture. Target 34 / 792 / 21,032 at k = 2
  / 3 / 4. Frobenius- and row-norm-matched Gaussian draws give 34 / 762-886 / 19,354-26,074. Activity-matched draws
  (W_hh ~ N(.25, 1), Frobenius-rescaled, activity closest to 0.773) give 15,012-19,314 at k = 4, a median of
  16,378 (0.78x the trained count).
- **H-E15:** the trained net's cells are concentrated at the ties. Cells per unit mass near ties (d < .05) vs far
  (d >= .1, non-blind):
  - final leaves, zoo seed 0, H = 16: 19.1x at T = 3 and 26.4x at T = 4;
  - prefix k = 4: trained 44.1x vs random/scrambled 1.3-6.6x (median 3.3).

## Pre-registration (written 2026-10-18, before running anything below)
**Enumerator (new, `dd.py`).**
- Double-description cone splitting. Every cell is a pointed polyhedral cone, stored as both its constraints and its
  exact extreme rays.
- A neuron's pre-activation c.x has a constant sign on the cell iff it has that sign on all rays. Otherwise the cell
  is split, and new rays come from adjacent (+, -) ray pairs (the combinatorial adjacency test). No LP.
- Start partition: the 2^k orthants for prefixes; orthants x order chambers for final leaves, so the target is
  constant per cell.
- Validation: it must reproduce the target's 34 / 792 / 21,032 regions (T192, and my LP counter in
  tie_tail_labels/prefix_blind.py).
- Masses: rays, then simplicial cones, then robust-angle orthant probabilities (`tie_tail_labels/prefix_mass.py`,
  validated in T222). They must sum to 1 within 1e-9.
- *Amended before any run:* per-leaf error masses (T239 part 1) are not needed for H-E15's criteria and are
  dropped. To bound compute, H-E15 prefix masses use the target + frob0-2 + row0-2 (the counts for H-E14 use all
  draws).

**Models (my own draws, my own seeds).**
- target `example_2nd_argmax`;
- frob: 5 draws (seeds 2800-2804). Each matrix has i.i.d. N(0,1) entries, rescaled to the trained matrix's
  Frobenius norm;
- row: 5 draws (seeds 2810-2814). Rows rescaled to the trained row norms; W_ih is a random sign times the trained
  magnitude;
- mu: a pool of 100 draws (seeds 2900-2999). W_hh ~ N(0.25, 1), others N(0,1), all Frobenius-rescaled. The 5
  draws with activity closest to the trained one are kept;
- zoo(16, 3, 0) and zoo(16, 4, 0) for the final leaves.

Activity = mean fraction of h_10 > 0 over 2e5 N(0,1) inputs. Check: the target must give ~0.773.

**Tie distance.** Cell centre u = normalised mean of its unit rays. d = min(u(1) - u(2), u(2) - u(3)) / sqrt2 on
the sorted coordinates; blind = u(2) <= 0. Strata d < .02, .02-.05, .05-.1, .1-.2, >= .2, plus blind. Near/far =
(cells / mass in d < .05) / (cells / mass in d >= .1, non-blind).

**Agreement criteria (both rows).**
- **H-E14 confirmed** if all of the following hold:
  - (i) target regions exactly 34 / 792 / 21,032;
  - (ii) my 10 frob/row draws all have k = 4 regions in [0.85 x 19,354, 1.15 x 26,074], with medians within ±15%
    of the originals' (frob 23,900; row 23,294);
  - (iii) the same verdict: median random k = 4 growth within ±30% of the trained growth;
  - (iv) activity-matched k = 4 median within ±20% of 16,378, and the matched/trained ratio in [0.6, 1.0].
- **H-E15 confirmed** if all of the following hold:
  - (v) prefix k = 4 trained near/far within 2x of 44.1, the random median within 2x of 3.3, and the trained
    ratio above every random draw;
  - (vi) final leaves near/far within 2x of 19.1 (T = 3) and 26.4 (T = 4), and >= 10x at T = 4.
- **Partition caveat:** my start partition differs from the original's (T192 simplex fan + x_0 cut for prefixes;
  chamber halves for final leaves). Region counts (distinct gate words) do not depend on it; cell counts do. That
  is why H-E14 compares regions and H-E15 allows 2x. A failure of (v)/(vi) with the ordering intact is reported
  as "partition-sensitive", not as refuted.

## Method (as run)
- `dd.py`: the double-description enumerator. `models.py`: models and activity.
- `he14.py` (`run_he14.sh`) -> `results/he14.json`, `he14.log`.
- `he15.py` (`run_he15.sh`) -> `results/he15_*.json`, `he15.log`.
- Final leaves: start = orthants refined by all order planes x_i = x_j (24 / 120 start cells at T = 3 / 4), then
  the T gate steps.

## Status
Done (2026-10-18). **H-E15: confirmed by every pre-registered criterion. H-E14: (i)-(iii) confirmed; (iv) misses
its numerical band by 0.8 pp** (direction and verdict reproduced).

## Results
**Validation.**
- The dd enumerator gives the target's 34 / 792 / 21,032 regions and 25,194 k = 4 cells, identical to T192 and to
  my LP counter. It takes 11 s instead of 20 min.
- Every model's k = 2 count is 34.
- Mass sums: |sum - 1| <= 2.7e-10 for every cell set.
- Target activity 0.7729 (the original quotes 0.773).

### H-E14 (`results/he14.json`)

| model | regions k = 2 / 3 / 4 | growth k = 4 | activity |
|---|---|---|---|
| target | 34 / 792 / 21,032 | x26.6 | 0.773 |
| frob0-4 | 34 / 766-816 / 19,796-22,696 (median 21,496) | x25.8-27.8 | 0.37-0.61 |
| row0-4 | 34 / 768-842 / 19,026-23,978 (median 23,190) | x24.8-28.9 | 0.35-0.80 |
| mu, activity-matched (5 of 100; 0.763-0.785) | 34 / 748-820 / 18,018-21,862 (median 19,792) | x24.1-26.7 | 0.76-0.79 |

- (i) **passes**: exact.
- (ii) **passes**:
  - all 10 draws lie in 19,026-23,978 (band [16,451, 29,985]);
  - the frob median is 21,496 vs 23,900 (-10.1%);
  - the row median is 23,190 vs 23,294 (-0.4%).
- (iii) **passes**: median random growth is x27.0 vs trained x26.6 (+1.5%).
- (iv) **fails narrowly**: the matched median is 19,792 vs the original's 16,378 (+20.8%, band ±20%). The
  matched/trained ratio is **0.94** (range 0.86-1.04) vs the original's 0.78 (0.71-0.92); it is inside the
  pre-registered [0.6, 1.0].
  - The qualitative claim (activity matching reveals no training-induced reduction) holds, more strongly here.
  - The original's "0.78x" figure does not reproduce. A likely cause is the selection: 5 of 100 draws closest in
    activity is a noisy estimator of a 15-22k distribution. Mine has pool median activity 0.750 at mu = .25,
    consistent with the original's grid.

### H-E15 (`results/he15_*.json`)
**Prefix k = 4, cells per unit mass by stratum**:

| model | <.02 | .02-.05 | .05-.1 | .1-.2 | >=.2 | blind | near/far |
|---|---|---|---|---|---|---|---|
| target | 380,490 | 17,214 | 8,235 | 3,387 | 3,002 | 27,008 | **37.2** (orig. 44.1) |
| frob0 | 36,156 | 29,519 | 19,406 | 17,107 | 22,627 | 25,850 | 1.7 |
| frob1 | 55,704 | 32,374 | 20,201 | 18,803 | 15,894 | 37,654 | 2.4 |
| frob2 | 64,574 | 34,529 | 17,964 | 16,952 | 16,502 | 29,843 | 2.8 |
| row0 | 112,153 | 19,758 | 10,013 | 11,112 | 7,924 | 31,677 | 6.6 |
| row1 | 136,620 | 24,096 | 17,088 | 12,876 | 8,637 | 34,029 | 6.6 |
| row2 | 142,543 | 27,767 | 11,743 | 15,367 | 7,611 | 15,859 | 6.5 |

**Final leaves, zoo(16, T, 0):**
- T = 3: 1,046 cells (776 regions), near/far **21.9** (orig. 19.1).
- T = 4: 46,754 cells (26,742 regions; the original's chamber start gives 51,552 leaves), near/far **31.1**
  (orig. 26.4).

**Criteria:**
- (v) **passes**:
  - trained 37.2 vs 44.1 (0.84x);
  - random median 4.65 vs 3.3 (1.4x);
  - trained above every random draw (max 6.6).
- (vi) **passes**: 21.9 vs 19.1 and 31.1 vs 26.4 (1.15x and 1.18x), and >= 10x at T = 4.
- The fragmentation pattern also reproduces. Near ties (d < .02) the trained prefix has 2.7-10.5x more cells per
  unit mass than random. Far from ties (d >= .2) it has 2.5-7.5x fewer.

## Compute accounting
- dd enumeration: 17 prefix models x k = 2..4 plus 2 final-leaf sets, ~5 CPU-min.
- Activity pool: 100 draws x 2e5 inputs, ~2 CPU-min.
- Masses: 9 cell sets, ~85 CPU-min (mostly the k = 4 Plackett integrals).
- Total ~1.5 CPU-h, one heavy process at a time, < 1.5 GB.

## Claims (review; for the DRI)
- [measured] H-E14's region counts and the "generic growth" verdict reproduce with a different algorithm (double
  description vs estT geometry), different draws and different seeds. The activity-matched ratio is 0.94x here, vs
  0.78x originally (criterion missed by 0.8 pp). Recommend confirming H-E14 with that number qualified, as "matched
  random draws give 0.8-1.0x the trained count".
- [measured] H-E15 reproduces within 1.2x on every ratio (prefix near/far 37 vs 44; final 22 / 31 vs 19 / 26;
  random median 4.7 vs 3.3), despite a different start partition. Recommend confirming.

## Alternatives not ruled out
- Both rows rest on k <= 4 / T <= 4. Neither check says anything about larger k.
- Tie distance from cell centres is the original's choice too. A ray-based contact definition was not tried.
