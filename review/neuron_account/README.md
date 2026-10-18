# Independent review of T241 (quantitative neuron account of M_16,10)

↑ [T241 README (branch claude/t241-neuron-account @a9b09a14)](../../m16/approaches/neuron_account/README.md) ·
[TASKS](../../TASKS.md) · [INDEX](../../INDEX.md)

## Question
Do T241's headline numbers and claims hold up under different code, seeds and RNG? The DRI asked for four checks:
1. the all-16 replacement agreement (.9554), the single-neuron replacement costs, and D1 (78.5% of disagreements
   in the 19.6% of mass near ties or in the blind region);
2. that the fits are genuinely held out;
3. B2 (n5/n9 are delayed comparators, R^2 .998/.9999) and A4 (the period-4.32 mode n3/n13/n15 is the most
   damaging mode);
4. C2 (99% agreement from unstructured noise needs 1 - R^2 ~ 1e-5).

## Method
- **Weights:** ARC's public checkpoint `gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth`,
  downloaded fresh. sha256 = 498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb, which matches
  T241's 498bd9a8... `rv.py` asserts this hash on every run.
- **Code:** my own float64 forward pass (`rv.states`). Inputs come from the Philox generator, seeds 11/12/13
  (train/val/eval) and 21-23 (checks); T241 used PCG64 seeds 2410xx.
- **Families:** written from T241's README definitions of the cells and bases (A1, A12, A123, A123K, H1, H3,
  Hall, A12K-Hall, comparator hinges = ReLU of the comparators' A123K formulas). I read `encode.py` only to settle
  the details the README leaves open: rank-absent age = t, K capped at 3, ridge 1e-6 relative.
- **Fits:** per-cell normal equations by sorting rows (T241 uses bincounts), with a pooled fit for unseen cells.
  The family is chosen per (neuron, step) by validation R^2, and everything is reported on the eval set. Same
  sizes as T241: 1M / 200k / 200k.
- **Scripts:**
  - `replace.py` (`run_replace.sh`) -> `results/replace.json`: replacement test, single neurons, cumulative
    groups, D1 stratification, row-overlap check.
  - `checks.py` -> `results/checks.json`: B2 from the exact weights, A4 with random-subspace and readout-direction
    controls, C2.

## Status
Done (2026-10-18). Every claim checked is reproduced in substance. There are three qualifications (C1, B2, A4)
and one framing issue for C1/C2.

## Results

### (1) Replacement test (`python replace.py 1000000 200000 200000`, ~5 CPU-min)

| quantity | T241 | this review |
|---|---|---|
| network accuracy on eval | .95247 | .95325 |
| **all-16 agreement** (best family incl. hinges) | **.9554** | **.9659** |
| all-16 accuracy - network | +0.11 pp | **+0.82 pp** |
| all-16 agreement without hinge families | .9425 | .9602 |
| single-neuron agreement: n4 / n9 / n2 | .998 / .997 / .995 | .997 / .997 / .995 |
| single-neuron agreement: n12 / n0 / n10 | .975 / .976 / .976 | .975 / .976 / .979 |
| single-neuron agreement: others | .978-.983 | .978-.985 |
| cumulative: n4 / +n2 / +n5,n9 / +cmp / +n3,n13 / +clocks | .998 / .994 / .979 / .961 / .956 / .955 | .997 / .993 / .981 / .967 / .962 / .966 |
| held-out R^2 at t=10, chosen family: n3 / n13 / n15 | .9854 / .9876 / .9943 | .9857 / .9879 / .9943 |
| D1: mass near tie (gap < .05) or blind | 19.6% | 19.5% |
| **D1: share of disagreements in that mass** | **78.5%** | **84.6%** |
| disagreement on network errors / on network-correct | 52.9% / 2.0% | 50.1% / 1.1% |
| blind: mass, disagreement rate | 1.1%, 45.8% | 1.06%, 52.7% |

- **Single-neuron replacement costs and per-neuron R^2 replicate closely,** within about 0.003 in agreement and
  0.0003 in R^2.
- **The all-16 headline replicates in substance.** Both versions are far below the pre-registered 99%, so C1
  ("target missed") stands. But the value depends on the implementation: mine is 1.0 pp higher in agreement
  (1.8 pp without hinges) and 0.8 pp higher in accuracy than the network. With 200k eval rows the agreement s.e.
  is about 0.04 pp, so this is a real implementation effect, not sampling. Candidate causes, not separated:
  pooled fallback and ridge details in sparse cells, and the order of float32 storage.
  - Read .9554 as "about 95.5-96.6%, implementation-dependent".
  - The accuracy result ("within 0.1 pp", +0.11) is fragile: my replacement is 0.8 pp *more accurate* than the
    network. That strengthens T241's D2 reading (the formulas describe a cleaner algorithm than the network)
    but means the +0.11 number is not robust.
- **Framing point (both versions).** With all 16 neurons replaced, h_10 = ReLU(f_10(x)) and no recurrence is
  used. The all-16 agreement is therefore a test of the **t = 10 formulas only**. T241's own json confirms this:
  `best` all-16 = `best_final_step_only` = .955395, and mine is identical too. Earlier-step formulas enter only the
  partial replacements. The README's claim that the account works "at every step" is supported by the single
  and partial replacements, not by the all-16 number.
- **D1 replicates in direction and size** (the 19.5% of mass holds 78-85% of the disagreements).

### (2) Held-out discipline
- **Code audit of T241** (`phaseB.py`, `phaseC2.py`, `encode.py`):
  - train, val and eval come from separate seeds (241031/241032/241033; Phase B 241021/241022);
  - the hinge bank is fit on train only;
  - the family is chosen on val;
  - R^2 and agreement are reported on eval;
  - the pilot means in Phase A are on a disjoint seed.

  I found no input reuse.
- **In my replication** the three sets have 0 shared rows (exact-byte hashes), and held-out R^2 matches T241's to
  3e-4. Overfitting would show up as a gap between the two codes' eval R^2, and there is none.
- **Remaining caveat (inferred):** "best family by val R^2" picks among 12 families per (j, t). On 200k val rows
  that selection bias is negligible next to the R^2 differences involved.

### (3) n5 / n9 and the modes (`python checks.py`, ~6 CPU-min)
**B2, n9: true, but it is the weight matrix, not a fitted discovery.** Exactly, pre_t[9] = Wh[9] . h_{t-1} + Wi[9]
x_{t-1}. The row Wh[9] is −5.03 on n7, −1.09 on n8, −0.36 on n1, −0.09 on n6, −0.55 on n9 (self), +0.15 on n4,
and |w| < 0.11 elsewhere. T241's "n9 ≈ −5.00 n7_{t−1} − 1.08 n8_{t−1} − 0.36 n1_{t−1}" is this row. With the exact
weights and no fit, the comparator columns alone give R^2 .9992-.9996 at every t = 2..10, and least squares on the
true lag-1 comparator states plus x_t gives .9997. So "n9 is a delayed comparator" is correct and can be read off
Wh directly.

**B2, n5: only partly supported.**
- Wh[5] = +1.19 n4, −0.85 n8, −0.59 n7, −0.42 n5, +0.33 n11, +0.30 n9, +0.26 n12, +0.22 n0, +0.21 n14, −0.20 n3.
- With exact weights, the comparator columns explain R^2 .67 at t=2 but only .36 at t=10 (.43 with n4 added).
- Least squares on the true lag-1 comparator states plus x_t: .73 at t=10.
- T241's .998 (family H1) comes from the A123K cells and base *plus* the hinges. The comparators are a large
  minority of n5's input, not its definition. "Delayed comparator (weak)" is fair; "both are delayed comparators,
  R^2 .998/.9999" overstates n5.

**A4, modes of final gate pattern 64975** (29.0% of final states; my own eigendecomposition and donors). Accuracy
inside the pattern after resampling one mode at h_10:

| mode | 1.010 (integrator) | 0.973 | 0.951, **8.42** | 0.937, **4.32** | 0.928, Nyquist | 0.837, 2.89 | 0.735, **8.37** | 0.231 | ~0, input |
|---|---|---|---|---|---|---|---|---|---|
| T241 | .905 | .842 | .864 | .491 | .824 | .553 | **.390** | .807 | .838 |
| review | .904 | .840 | .864 | .494 | .824 | .553 | **.390** | .807 | .836 |
| energy share | .453 | .115 | .311 | .063 | .007 | .044 | .042 | .006 | .290 |

- The table reproduces to 0.003.
- **The period-4.32 mode is NOT the most damaging. The damped period-8.37 mode is (.39 vs .49),** in pattern 64975
  and also at h_9 in T241's own table. T241's claim A4 correctly names both. The phrasing "4.32 is the most
  damaging mode" is wrong.
- **Neither mode is specially damaging against a control.** Resampling a *random* 1-D subspace of the active
  coordinates gives mean accuracy .452 (min .297, mean energy .070). A random 2-D subspace gives .304 (min .206,
  energy .158). The single readout-most-sensitive direction gives .587.
- So mode resampling mainly measures how sensitive the readout is to *any* incoherent perturbation, consistent
  with T241's own A5 caveat. Per unit energy (accuracy drop in pp per percentage point of energy share), the 8.37 mode (13.4) stands out
  somewhat. The 4.32 mode (7.4) is at the level of a random 1-D direction (7.2).
- Patterns 63951 and 65007 show the same pattern (random 2-D ~ .30; mode drops 15-70 pp). In 65007 the oblique
  projector energies are 1.4-44x the state energy, confirming T241's ill-conditioning caveat.

### (4) Noise injection (C2)
Gaussian noise on the true pre_10 with s.d. sqrt(eps) x s.d.(pre_10[j]), on 200k fresh inputs:

| 1 - R^2 | 1e-2 | 3e-3 | 1e-3 | 3e-4 | 1e-4 | 3e-5 | 1e-5 | 3e-6 |
|---|---|---|---|---|---|---|---|---|
| T241 | .612 | .770 | .875 | .938 | .967 | .982 | .990 | - |
| review | .610 | .769 | .874 | .939 | .967 | .982 | **.990** | .994 |

- **Reproduced: 99% needs 1 - R^2 ≈ 1e-5 on every neuron** for unstructured errors.
- Testing pre_10 only is the right test for the all-16 case, which depends on t=10 alone (point (1)).
- The argument's own caveat holds: real formula errors are correlated and readout-aligned to a lesser degree, so
  95.5-96.6% is reached at 1 - R^2 ~ 3e-3.
- R^2 here counts variance on the inactive (negative) side, where errors are mostly harmless after the ReLU, so
  the 1e-5 figure is a conservative bound.

## Compute accounting
- Forward passes (1.6M training/val/eval sequences plus 800k for the checks) and per-cell least squares.
- replace.py: ~5 CPU-min, peak below the 2.5 GB cap after one out-of-memory fix. checks.py: ~6 CPU-min.
  Smoke tests: ~1 CPU-min.
- Total about 15 CPU-min, single thread.

## Claims (review; unverified until the DRI accepts)
- [measured] Weights sha256 498bd9a8... confirmed. Single-neuron replacement agreements, held-out R^2, the mode
  table and the noise curve all reproduce within 0.003 with independent code and RNG.
- [measured] All-16 agreement is .9659 here vs .9554 in T241. The 99% target is missed in both, but the value
  depends on the implementation at the ~1-2 pp level. The accuracy gap is +0.82 pp here vs +0.11 pp.
- [measured] The all-16 number tests only the t=10 formulas: all-16 = final-step-only, exactly, in both codes.
- [measured] No train/val/eval overlap (0 shared rows here; separate seeds and fit/selection/report split in
  T241's code).
- [measured] n9's "delayed comparator" coefficients are Wh[9]. The exact comparator columns give R^2 .9992 with
  no fit. n5's comparator share is only .36-.73 at t=10.
- [measured] The most damaging mode in pattern 64975 is the damped 8.37 mode (.39), not the 4.32 mode (.49). A
  random 2-D subspace resample is more damaging than either on average (.30).

## Alternatives not ruled out
- The 1-2 pp all-16 discrepancy could come from cell sparsity handling (ridge, pooled fallback), from float32
  storage of formulas, or from a family-definition detail I reconstructed differently. I did not diff against
  T241's saved eval file. Running T241's code on my eval inputs, or mine on theirs, would separate these.
- The random-subspace control uses orthogonal projectors and is not energy-matched. An energy-matched control
  could change the per-energy ranking.

## Failure modes
- If the cell definitions (absent-age coding, K cap) differ from T241's in a way I did not catch, the agreement
  difference would be a definition effect, not robustness. Held-out R^2 matching to 3e-4 argues against a large
  definition gap.
