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
