# tie_tail_labels (T218): is the heavy tail of the tie-slicing integrand a few mechanism-labelled families?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md) ·
[sci_tao (E25)](../sci_tao/README.md) · [ht_seeds (E33)](../ht_seeds/README.md)

## Question
Deterministic tie-slicing (E25/E33) loses its edge at T~5-8 because the face integrand is heavy-tailed (top 5% of
tie points carry ~35% or more of the second moment, H-E12). Direction D2 asks whether that tail is made of a FEW
mechanism-labelled families. If it is, each family could get a dedicated deterministic rule.

## Pre-registration (written 2026-10-18, before any measurement)

**Integrand.** sci_tao's exact co-area identity, copied unchanged (`core.py`): err = Z23 E23[F] + Z12 E12[F].
With n faces per kind, the per-point contribution is g = Z_kind * F. Tail = the top 5% of all 2n points by g^2.
Tail mass = their share of sum g^2.

**Primary label (L_mech)**, computed per tail point from an exact re-walk of its segment:
(kind, positions (a, b, p1) [these fix the pair distance |a-b| vs clock period, the age of the max, record order and
the x_{T-1} case], value bucket of the 2nd-largest value u (<=0 = blind, 0-0.25, 0.25-0.5, >0.5), wrong-answer
type (tie partner / max / 3rd / other), and the set of (step, neuron) gate flips strictly inside the dominant
misclassified interval (up to 3; "none" = the band sits in one linear cell)).
Coarser schemes are reported too (positions only; positions+flip set; mechanism without positions). At T=6 there
are only 75 position configurations, so any scheme coarser than L_mech passes "<=100 labels" trivially. Those
schemes are descriptive and do not decide.

**Primary metric: out-of-sample coverage.** Choose the top-L labels by tail mass on rule A (Sobol seed 700, 2^14
faces per kind). Then measure the share of rule B's tail mass (seed 701, independent scramble) that falls in those
labels, for L = 10, 30, 100. This guards against the trivial in-sample case where every label is a single point.

**Decision rule** (L_mech, out-of-sample, L = 100):
- **Go**: coverage >= 90% on every model: zoo (16,6) seeds 0-4, target M_16,10, and noisy copies sigma 0.005 and
  0.01 (sigma x matrix RMS, seed 0).
- **Kill**: coverage < 70% on the target or on either noisy copy.
- Otherwise **inconclusive**, with what would separate the cases.

**Controls.**
- Singleton null: top-L individual points (what any too-fine labelling achieves in sample).
- Random-label null: the same label-size profile with labels randomly reassigned.
- Sample-size check: in-sample share at 2^13 vs 2^14 faces. Real families should give a share that is stable in n.

**Alternative explanations to separate (AGENTS rule 3).**
1. *Few families* (the D2 hope): coverage is high out of sample and stable in n, the same labels lead across noisy
   copies, and the integrand within a label is light-tailed.
2. *Continuum of rare events*: the tail is a spread of near-unique (position, value, flip) combinations. Then
   coverage is low out of sample and the in-sample share falls as n grows.
3. *Few coarse families that are still heavy-tailed inside*: labels cover the mass but the within-label tail
   share stays near the global one. Then the labels would not give a usable dedicated rule. Test: the within-label
   top-5% share for the top labels.
4. *Model-specific labels*: concentration holds on each model, but the leading labels differ between orig and its
   noisy copies, so no rule is fixed across copies. Test: overlap of top-100 label sets across copies.

**Blind sub-test.** Exact prefix region counts of the target restricted to {at most 1 positive coordinate}, k = 1,
2, ... Go if the region growth is <= 4x per step.

## Method
- `core.py`: sci_tao's co-area core, copied unchanged (face laws, Sobol rule, exact vectorised walk).
- `tail.py MODEL`: for rule seeds 700 (A) and 701 (B), 2^14 faces per kind, computes g = Z*F for every face, takes
  the top 5% by g^2 (1639 points), and re-walks each tail segment with a scalar exact walk (`walk_one`). That walk
  reproduces the vectorised F to <= 9e-12 relative on every tail point (`Fcheck` in the json). It records the
  misclassified components, the class predicted there, and the (step, neuron) gate flips strictly inside each
  component. Writes `results/MODEL.json`.
- `analyse.py` -> `analyse.out`, `summary.json`: label schemes, in-sample and out-of-sample coverage, nulls,
  n-check, within-label tail share, composition, cross-copy coverage.
- `blind_split.py` -> `results/blind_split.json`: second moment split by blind faces (2nd-largest value u <= 0)
  over ALL faces (rule A), plus the tail share of the non-blind remainder.
- `prefix_blind.py` -> `results/prefix_blind*.json`: an independent LP cone-splitting region counter (scipy HiGHS;
  T192's prefix.py needs its geometry stack). Validated on full R^k against T192: k=2 gives 34 regions and k=3
  gives 792, identical to T192. Run restricted to the k+1 cones with at most one positive coordinate.
- Models: `alg_zoo.zoo_2nd_argmax(16, 6, seed=0..4)` (z6s0-4); target `example_2nd_argmax()` (orig); noisy copies
  sigma 0.005 and 0.01 (sigma x matrix RMS, rng seed 0, first copy = leaderboard `s0.005_0`; error 10.8% matches
  its truth 0.8918). Error estimates from the faces match the truths (z6s0 2.16% vs 2.17%; orig 4.66% vs 4.69%).
- Reproduce: `./run_all.sh` (8 models, ~2.5 min total), `python analyse.py > analyse.out`,
  `python blind_split.py`, `./run_blind.sh` (blind prefixes), `python prefix_blind.py 3 --full` (validation).
  Environment: single thread, `ulimit -v 2500000`.

## Status
Done (2026-10-18). Verdict under the pre-registered rule: **KILL** for direction D2 as posed (negative, unverified).

## Results

### Main table (rule A = Sobol seed 700; oos = labels chosen on A, coverage of rule B's tail mass)

| model | err | top5% share | blind g2 share | top5% non-blind | mech oos L10/30/100 | mech in-sample half / A / A+B (L100) | pos oos L100 (#labels) | nopos oos L100 | coarse oos L10/L30 | null singleton L100 | null random L100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| z6s0 | 2.16% | 0.83 | 0.79 | 0.62 | 0.017 / 0.074 / 0.112 | 0.61 / 0.52 / 0.45 | 0.98 (119) | 0.36 | 0.81 / 0.98 | 0.47 | 0.069 |
| z6s1 | 2.98% | 0.84 | 0.86 | 0.59 | 0.013 / 0.054 / 0.144 | 0.67 / 0.56 / 0.45 | 0.96 (120) | 0.46 | 0.85 / 0.98 | 0.51 | 0.067 |
| z6s2 | 1.64% | 0.84 | 0.83 | 0.60 | 0.127 / 0.191 / 0.308 | 0.76 / 0.69 / 0.62 | 0.99 (117) | 0.51 | 0.87 / 0.98 | 0.57 | 0.081 |
| z6s3 | 1.54% | 0.90 | 0.87 | 0.62 | 0.083 / 0.162 / 0.241 | 0.74 / 0.65 / 0.58 | 0.99 (113) | 0.55 | 0.90 / 0.99 | 0.58 | 0.063 |
| z6s4 | 1.70% | 0.88 | 0.83 | 0.74 | 0.024 / 0.048 / 0.156 | 0.64 / 0.55 / 0.44 | 0.98 (118) | 0.38 | 0.78 / 0.98 | 0.49 | 0.057 |
| **orig** | 4.66% | 0.87 | 0.60 | 0.72 | **0.008 / 0.009 / 0.023** | 0.68 / 0.54 / 0.42 | 0.35 (498) | 0.30 | 0.70 / 0.95 | 0.54 | 0.063 |
| **s0.005** | 10.83% | 0.72 | 0.25 | 0.67 | **0.006 / 0.007 / 0.043** | 0.47 / 0.34 / 0.25 | 0.54 (456) | 0.19 | 0.54 / 0.91 | 0.32 | 0.044 |
| **s0.01** | 24.26% | 0.64 | 0.08 | 0.64 | **0.004 / 0.012 / 0.033** | 0.39 / 0.27 / 0.19 | 0.73 (391) | 0.09 | 0.73 / 0.94 | 0.26 | 0.036 |

Schemes: mech = (kind, a, b, p1, u-bucket, wrong type, in-band flips), the primary label. pos = (kind, a, b, p1).
nopos = (kind, u-bucket, wrong type, in-band flips). coarse = (kind, u-bucket, wrong type, min(#flips, 3)).
Also in `analyse.out`: pos_flips, and relpos (pair distance, age of max, x_{T-1} case instead of absolute
positions). On the target relpos gives 0.044 oos at L100, so relative positions do not help.

**Reading.**
1. **Primary rule: KILL.** On the target and both noisy copies, the top-100 mechanism labels cover 2-4% of an
   independent rule's tail mass, far below 70%. That is below the random-label null (4-6%), because most mech
   labels are singletons. It fails on (16,6) as well (11-31%).
2. **The tail is a continuum of rare events (alternative 2), not a few families.** In-sample "concentration"
   (42-68% at L=100) is the singleton effect: it matches the top-100-points null (0.26-0.58), and it FALLS as n
   grows (half-sample > A > A+B on every model). Real families would give a share that is stable in n.
3. **Coarse labels cover but do not regularise (alternative 3).** The coarse scheme holds 91-99% at L=30, out of
   sample, on every model. But within its top labels, the top 5% of points still carry 18-46% of the label's tail
   mass, against 43-54% pooled. On the target that is 0.18-0.46 vs 0.49. The tail is about as heavy inside a
   family as outside it.
4. **Position configurations** pass trivially at T=6 (about 120 configs, so <= 100 labels covers 96-99%). At T=10
   (about 400-500 configs) they reach only 35-73%.
5. **Labels are not fixed across copies (alternative 4).** Orig's top-100 labels (from rule A) cover rule B's tail
   mass on the copies as follows:

   | scheme | orig | s0.005 | s0.01 |
   |---|---|---|---|
   | mech | 2.3% | 0.3% | 0.0% |
   | pos | 35% | 28% | 26% |
   | nopos | 30% | 15% | 3% |

   The tail also changes kind with noise. Blind faces are 60% of the second moment on orig, 25% on s0.005 and 8%
   on s0.01, where 65% of the tail sits at u > 0.5 instead.

### What the tail IS made of (descriptive; rule A tail-mass shares, `analyse.out` "comp")
- **The blind region is the single largest family.** It is 1.7% of faces at T=10 but carries 60% of the full second
  moment (79-87% at (16,6), where it is 13.8% of faces). In the tail the network mostly answers a non-tie position
  ("other" 49% on orig). This matches FINDINGS' "nearly blind, wrong 78%".
- Removing the blind faces does **not** leave a light tail: the non-blind top 5% still carries 59-74% of the
  non-blind second moment (orig 0.72; top 1% = 0.44).
- In-band gate flips on orig are dominated by **neuron 5** at steps 5-9: (7,5) 12%, (9,5) 8%, (6,5) 7%, (8,5) 7%,
  then n3. n5 is the "x8 near-record" neuron (FINDINGS sec. 2). 27% of orig tail mass has >= 5 in-band flips, 20%
  has none. Wrong components are mostly on the t<0 side (86%) and are single components (92%).
- Pair distance |a-b|, age of max and the x9 case are spread out: there is no peak at distance ~8 (the clock
  period). Distance 8 carries 3.5% of orig tail mass and distance 1 carries 22%, roughly monotone, as expected from
  the number of pairs at each distance.

### Blind sub-test (`results/prefix_blind.json`)
Exact activation regions of the target's k-step prefix, restricted to inputs with at most one positive x_i
(i < k). For comparison, the unrestricted counts are T192's, which my counter reproduces at k=2 and 3.

| k | blind regions | growth | full-space regions (T192) | growth |
|---|---|---|---|---|
| 1 | 2 | - | 2 | - |
| 2 | 25 | 12.5x | 34 | 17x |
| 3 | 406 | 16.2x | 792 | 23x |
| 4 | 7,687 | **18.9x** | 21,032 | 26.6x |

**No-go** (threshold <= 4x per step). Inside the blind region the prefix still branches 12-19x per step, and the
growth rate is rising, approaching the full-space rate (~25x). Being blind to magnitude rank does not make the
network's gating simple there: the negative inputs still drive the Wi<0 neurons through many gate patterns. k=5 was
stopped for cost (estimated ~2.5M LPs, ~2 CPU-hours). Reproduce: `./run_blind.sh` (stop after k=4).

## Compute accounting
- Unit: one forward pass of the length-T sequence. Walk cost is counted as in sci_tao (2(T - restart)/T per cell).
- Tail runs: 8 models x 2 rules x 2^15 faces, about 1e6-1e7 pass-equivalents, plus tail re-walks (~3e5).
  Wall time ~2.5 min single thread.
- blind_split: 8 x 2^15 faces, ~1 min.
- Prefix LPs: 143k LPs (k <= 4, ~6.5 CPU-min). The full-space validation at k <= 3 took 16k LPs.
- Total under 1 CPU-hour. No sampling-based inputs: deterministic Sobol rules over faces (quadrature class).

## Claims (all unverified; for the DRI)
- [measured] On M_16,10 and its sigma 0.005 / 0.01 copies, the top-100 pre-registered mechanism labels cover
  2.3% / 4.3% / 3.3% of an independent rule's top-5% tail mass. **KILL** under the pre-registered rule.
- [measured] In-sample concentration of fine labels falls with sample size and matches the singleton null. The
  tail is a continuum of rare (position, value, flip) combinations at the resolution tested.
- [measured] Coarse mechanism families (<= 30 labels) cover >= 91% of the tail out of sample, but they stay
  heavy-tailed inside (within-label top-5% share up to 0.46 vs 0.49 pooled on orig).
- [measured] The blind region (2nd-largest <= 0) carries 60% of the target's face second moment from 1.7% of faces
  (79-87% at (16,6)). The non-blind remainder is still heavy-tailed (top 5% = 72% on orig).
- [measured] Leading labels are model-specific: orig's top-100 mech labels cover 0-0.3% of the copies' tails.
- [inferred] A dedicated deterministic rule per mechanism family would at best re-stratify the face integral. With
  the heavy tail inside each family it would not beat sampling by more than the existing 1.3-1.7x class.
- [measured] Blind sub-test no-go: blind-restricted prefix regions grow 12.5x, 16.2x, 18.9x per step (k=1..4).
- [speculative] Removing the blind region exactly (an exact or analytic treatment of its 1.07% mass) could still
  cut the face second moment by up to ~2.5x on orig (60% of it). That is a variance reduction, not a mechanistic
  estimator, and it only helps on low-noise copies (8% on s0.01).

## Alternatives not ruled out
- **Label resolution.** Families might exist at a resolution between "coarse" (about 60 labels) and "mech" (about
  2500), e.g. flips coarsened to neuron only (no step), or u-buckets finer than 4. Not swept. The within-label tail
  check suggests coarser labels do not regularise, but intermediate schemes were not tested.
- **Labels from other features.** Logit-margin geometry, the cell's linear map, or hidden-state clock phase were
  not used as labels. A family defined by a continuous feature (e.g. clock phase at the max) is not excluded.
- **Weighting.** The tail is defined by g^2 under the face law. A tail-aware importance rule (rather than labels)
  might still regularise the integrand. That is a different question (E26's per-triple sets, H-E11's budget
  dependence).
- **Statistical resolution.** 1639 tail points per rule. Families with fewer than ~5 expected points per rule
  cannot be seen as families. With 2^18 faces some fine labels might stabilise, although the falling-with-n trend
  argues against it.

## Independent review (T227, review/tie_tail_evidence on the shared branch)
T227 re-derived all 1,015 saved comparisons (26,224 tail points, 36 cross-copy scores) from the stored records,
with independent code and exact rational arithmetic. It confirms the primary fine-label L100 coverage of
2.339 / 4.295 / 3.276% (target / s0.005 / s0.01). That supports the **fixed-rule negative**. It does not establish
a population tail law or an impossibility for the estimator family. Caveats to carry with every use of Part 1:
- **The half-size n-check thins an already-selected tail.** It is not an independent 2^13-face run. The
  "falls with n" argument is therefore weaker than stated above.
- **Random-label null:** assignments are uniform over K labels and do not preserve label occupancy. They were not
  archived, and neither the half-sample RNG nor the null RNG was replayed (both are seeded `default_rng(0)` in
  `analyse.py`, in a fixed order).
- **Within-family moments condition on the global tail** (only tail points were stored). The within-label
  "heavy inside" check (alternative 3) is therefore not a proper full-family regularity test. That test is open as
  T230.
- **Blind grouping:** the face value u <= 0 is not the same as "<= 1 positive input" along the whole fibre (the
  segment moves x_a and x_b).
- **Stored data:** flip lists are ordered and may repeat. No full-face arrays, weight or runtime hashes, exact
  walk certificates or prefix history sets were stored.
- **Precision:** walk, LP and tie handling are in floating point.
- **Scope of the verdict:** the KILL applies to the pre-registered fixed fine-label rule. The coarse families
  (L30 coverage 91-99%) are not excluded as a basis for dedicated rules. Whether they regularise is T230.

## Failure modes
- Fine labels at this sample size are mostly singletons, so any labelling as fine as `mech` will look concentrated
  in sample. Only the out-of-sample and n-check numbers are informative.
- The "wrong type" and "flip" labels use the mass-dominant component only (single component in 92% of orig tail
  mass).
- The blind prefix counter uses an LP tolerance of 1e-9: pieces thinner than that are merged (same as any finite
  precision count). It is validated only up to k=3 on the full space.

---

# Part 2: T222 independent replication — mass coverage of the target's prefix regions (DRI-assigned 2026-10-18)

## Question
T217 found that the mass-weighted exp-entropy of gate words grows only ~x5 per step, while the region count grows
x25-30 per step. If the number of regions needed to cover 1 - delta of the Gaussian mass also grows slowly, then a
mass-ordered, best-first exact enumeration with exactly known uncovered mass could reopen the exact route.

## Pre-registration (written before computing any mass)
- **Regions:** the same LP cone-splitter as Part 1 (`prefix_blind.py --full`, all 2^k orthant start cones), k = 2,
  3, 4. Expect 34 / 792 / 21,032 regions (T192). Region = distinct gate history; its mass = the sum of its cells'
  masses.
- **Masses:** exact Gaussian mass of each polyhedral cone. Extreme rays come from the irredundant H-representation.
  Each cone is triangulated into simplicial cones. A simplicial cone with ray matrix R has mass P(lambda >= 0) with
  lambda ~ N(0, (R^T R)^-1):
  - k=2: arcsin closed form;
  - k=3: closed form 1/8 + sum asin(rho)/(4 pi);
  - k=4: Plackett's identity, a 1-D integral along R(s) = (1-s) I + s C, with closed-form 2-D conditional orthant
    probabilities, done by adaptive quadrature at tolerance ~1e-14.

  Deterministic throughout. Validation: masses sum to 1 within 1e-9 at every k; the Plackett code matches the
  closed form on k=3 cones; a sampling cross-check of per-region masses (1e6 samples) is diagnostic only.
- **Metric:** N_delta(k) = the number of regions, sorted by decreasing mass, needed to cover 1 - delta, for
  delta = 1e-2, 1e-3, 1e-4, 1e-5, 1e-6. Per-step growth g_delta(k) = N_delta(k) / N_delta(k-1).
- **Decision lines** (on delta = 1e-5, growth from k=3 to k=4, the last and most informative step):
  - **GO:** g <= 6x per step.
  - **DEAD:** g >= ~20x per step (near the ~25x count growth).
  - **Inconclusive** in between. Report the trend over k=2 to 3 to 4, and whether g rises with k.
- **Alternatives to keep in mind:**
  1. N_delta grows slowly at small k only because the regions are still few (saturation: N_delta <= count). Check
     N_delta / count.
  2. Slow growth at large delta but fast growth at small delta (a heavy tail of small regions), which would kill
     exact enumeration even if exp-entropy grows slowly. Compare g across delta.

## Method (part 2)
- `prefix_blind.py 4 --full --dump` (`run_full.sh`): the Part 1 LP cone-splitter on all 2^k orthant cones, cells
  pickled to `data/` (git-ignored).
- `prefix_mass.py` (`run_mass.sh`):
  - extreme rays by batched cofactor null vectors of all (k-1)-row subsets;
  - simplicial cones: fan for k=3, Delaunay of the section c.x = 1 for k=4;
  - orthant probabilities from **robust facet-normal angles** (2 atan2(|a-b|, |a+b|), so asin(C_ij) = pi/2 -
    theta_ij without cancellation): k=2 and k=3 in closed form, k=4 by Plackett's identity written per pair with
    rho = sin(phi). Gauss-Legendre with 48 and 96 nodes; where the two disagree by > 1e-13, a composite
    Gauss-Legendre graded geometrically toward the endpoint (45 levels x 12 nodes).
- Checks:
  - `check_orth4.py`: P4 vs a second Plackett path (from Y_4 decoupled), agreement 1e-16 on ordinary cones.
  - Graded rule vs breakpoint-adaptive quad: <= 5e-14 on cones with rays 1e-2 to 1e-7 apart. Plain quad and Genz
    (scipy mvn) are both off by up to 1e-8 there.
  - P4(rho = 1/2) = 1/5 exactly.
  - `check_mass_mc.py`: per-region masses vs 2e6 samples.
- First attempt (plain asin of (R^T R)^-1 correlations) lost ~7e-10 per thin simplex and gave a negative region at
  k=3. Replaced, and kept in the git history.

## Results (part 2) — `results/prefix_mass.json`, `run_mass.log`

| k | regions (T192) | mass sum - 1 | N_1e-2 | N_1e-3 | N_1e-4 | N_1e-5 | N_1e-6 | exp-entropy | N_1e-5 / regions |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 34 (34) | -9e-16 | 22 | 31 | 33 | 34 | 34 | 12.3 | 1.00 |
| 3 | 792 (792) | -3.5e-14 | 201 | 388 | 527 | 634 | 708 | 65.9 | 0.80 |
| 4 | 21,032 (21,032) | -2.7e-10 | 1,737 | 4,018 | 7,021 | 10,233 | 13,165 | 301.7 | 0.49 |

| step | g_1e-2 | g_1e-3 | g_1e-4 | **g_1e-5** | g_1e-6 | regions | exp-entropy |
|---|---|---|---|---|---|---|---|
| 2 -> 3 | 9.1 | 12.5 | 16.0 | 18.6 | 20.8 | 23.3 | 5.4 |
| 3 -> 4 | 8.6 | 10.4 | 13.3 | **16.1** | 18.6 | 26.6 | 4.6 |

The smallest k=4 region mass came out at -2.7e-11 (rounding). The k=4 mass sum is within the 1e-9 validation bound.
Sampling check: k=3 z-scores over 212 regions with mass > 1e-4 have mean -0.02 and s.d. 1.01 (max 2.65), and no
sampled gate word falls outside the exact regions. k=4: CHECK4.

**Verdict (pre-registered lines): NOT GO.** g_1e-5(3->4) = 16.1x, far above the 6x go line. It sits between go and
the ~20x dead line, so formally **inconclusive, leaning dead**.
- **Exp-entropy replicates T217.** It grows x5.4 and then x4.6 per step. But N_delta grows 3-4x faster than
  exp-entropy at every delta <= 1e-3. That is alternative 2: a heavy tail of small regions, so a slowly growing
  effective number of regions does not mean a slowly growing covering set.
- **Growth rises steeply as delta shrinks** (8.6x at 1e-2 up to 18.6x at 1e-6 at k=4). An exact method that must
  know the uncovered mass to ~1e-5 pays close to the full count growth.
- **Saturation (alternative 1) inflates nothing here; it deflates.** At k=2 every region is needed (N/regions = 1),
  so g(2->3) is capped by the count, and g(3->4) is the cleaner number. The slight fall from 18.6 to 16.1 may be
  partly this. N/regions falls from 0.80 to 0.49, so k=4 is the first unsaturated step.
- [inferred] Extrapolation, not measured: at ~16x per step from k=4, N_1e-5(k=10) ~ 1e4 x 16^6 ~ 2e11 regions.
  Even at 10x per step it would be ~1e10. Best-first exact enumeration does not look feasible on its own.
- What would separate inconclusive from dead: g_1e-5 at k=5 (~632k regions). With this LP code that is ~30-40
  CPU-hours; T192's geometry stack does it in ~0.5 CPU-hour, but its masses would be needed. If g(4->5) <= 10x,
  the route is still alive for a hybrid method (exact top regions plus a bounded remainder). If ~16x or more, it
  is dead.

## Compute (part 2)
- Enumeration: 414k LPs, about 21 CPU-min.
- Masses: k=4 had 146,714 simplices, nearly all on the graded rule, 11 CPU-min for k = 2-4.
- Checks: a few CPU-min. Single thread throughout.

## Claims (part 2, unverified)
- [measured] Region counts 34 / 792 / 21,032, identical to T192, with independent code.
- [measured] Exact Gaussian region masses sum to 1 within 2.7e-10 (k <= 4).
- [measured] N_1e-5 = 34 / 634 / 10,233. Growth 18.6x and 16.1x per step. Exp-entropy growth 5.4x and 4.6x.
- [inferred] Mass-ordered exact enumeration of prefix regions does not escape the count growth at delta ~1e-5.
