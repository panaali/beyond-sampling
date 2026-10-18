# T241: quantitative per-neuron account of M_16,10

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [FINDINGS](../../FINDINGS.md)

## Question
Can the qualitative account of the 16 neurons (running max n2, record detector n4, x9 comparators n1/6/7/8,
clocks n0/10/11/12/14/15, late-x8 n5/n9; FINDINGS.md, H-M1..H-M8) be made quantitative: a causal map, explicit
encoding formulas with held-out R^2, and a replacement model built from those formulas that reproduces the
network's answers (pre-registered targets: >= 99% agreement, accuracy within 0.1 pp, all 16 replaced)?

This is interpretability with sampling (not an ARC estimator). Every fit is scored on held-out inputs.

## Method
Exact float64 forward passes of the original checkpoint (`m16_weights.npz`, exported once from ARC's public
`.pth`, sha256 `498bd9a8...`; base accuracy on 1M fresh inputs 0.95332, truth 0.95318). `nacommon.py` holds the
forward pass, case masks and order statistics. Notation: h_t = state after reading x_{t-1}, t = 1..10.
Cases (by the true input): *x9max* (x9 is the max, 10.0%), *x9second* (10.0%), *x9below* (80.1%);
*blind* = at most one positive input (1.06%, overlaps the others).

- **Phase A, causal map** (`phaseA.py`, `phaseA_modes.py`, `reportA.py`): for each neuron j and step t,
  replace h_t[j] by 0 (zero), by its pilot-set mean (mean; pilot disjoint from eval) or by h_t[j] of another
  random input (resample), then run the remaining steps exactly. Also ablate neurons/groups at every step, or
  at the final state only. Modes: for the 5 most common final gate patterns, eigen-decompose D_g W_hh and
  resample each mode's oblique coordinates (from left eigenvectors) from a donor input with the same final
  gate pattern, at h_10 (readout) or at h_9 (then step 10 run exactly).
- **Phase B, encoding** (`encode.py`, `phaseB.py`, `reportB.py`): regress each pre-activation pre_t[j] on
  interpretable variables, all positively homogeneous of degree 1 like the network (no intercepts). A family is
  *cell x base*: inside each discrete cell (ages of the running top-k, number K of positive inputs) pre_t[j] is
  linear in a base vector (x_t, running top-k values m1..m5, sums of the positive / negative parts of the
  remaining inputs, clock phasors m_k cos/sin(theta a_k) with theta = 2 pi/8.42). Families in increasing size:
  L (x_t, m1..m3) · C (+clock phasors) · A1 (cell = age of max) · A12 (ages of top-2) · A123 (ages of top-3,
  base m1..m5) · A123K (+K, base + rest+/-) · A1234K · A1234K-all. Train 1M (seed 241021), held-out test 200k
  (seed 241022). Also per gate pattern (gates of h_9, top 3) and in the mode basis of final pattern 64975.
  After this run we added **comparator-hinge** families (`encode.HFAMILIES`, `HingeBank`): the base also gets
  c_{k,s} = ReLU(F_{k,s}(x)) for the four comparators k in {1,6,7,8} at the previous 1, 3 or all steps s, where
  F_{k,s} is the comparator's own A123K formula (held-out R^2 >= 0.999). These were scored in `phaseC2.py`.
- **Phase C, replacement** (`phaseC.py` v1 without hinges, `phaseC2.py` v2 with hinges, `reportC.py`,
  `phaseC_diag.py`): every neuron j at every step t gets a formula f_{j,t}(x) for its pre-activation, fit to the
  ORIGINAL network. A replaced neuron's state is h_t[j] = ReLU(f_{j,t}(x)), computed from the true input; the
  other neurons run the true recurrence on the partly replaced state. Formula = per (j, t) the family with the
  best R^2 on a validation set ("best"); "best_noH" = same, hinge families excluded. Train 1M (seed 241031),
  validation 200k (241032), evaluation 200k (241033), all disjoint; R^2 reported on the evaluation set.
  We replace single neurons, single groups, groups cumulatively (n4, n2, n5/n9, comparators, n3/n13, clocks)
  and neurons cumulatively (best first). Diagnostics: (1) i.i.d. Gaussian noise on the true pre_10 at
  R^2 = 1 - eps for every neuron (how precise must a formula be?); (2) fit the 10 logits directly from the same
  variables (is the limit the variables or the neuron split?).
- **Phase D, residual** (`phaseD.py`): stratify the disagreements of the all-16 v2 replacement by network
  correct/wrong, case, blind region, the global gaps m1-m2 and m2-m3, and how often the input's full 100-bit gate
  pattern occurs in the evaluation set.

## Status
Done (2026-10-18), all four phases. The pre-registered target was **not met**: with all 16 neurons replaced,
agreement is 95.5% (target >= 99%) and accuracy is 0.11 +- 0.05 pp above the network's (target within 0.1 pp).
All claims are unverified (no independent re-check yet).

## Results

### Phase A: causal map (N = 200,000 eval inputs, seed 241001; pilot 50,000, seed 241002)
Reproduce: `python3 phaseA.py 200000 && python3 reportA.py > results/phaseA_tables.md` (about 4 CPU-min);
`python3 phaseA_modes.py` (N = 400,000, seed 241011; about 1 CPU-min). Full 16x10 matrices for every metric
(accuracy, agreement, four case accuracies) x three ablation types are in `results/phaseA.json` and
`results/phaseA_tables.md`. Sampling s.e. of an accuracy here is about 0.05 pp; differences are paired.

**Resample-ablation of one neuron at one step: drop in accuracy (pp).** Base accuracy 0.9539.

| neuron | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 | t=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| n0 | -0.0 | 2.3 | 5.6 | 8.3 | 14.9 | 23.3 | 28.0 | 35.6 | 37.2 | 42.5 |
| n1 | 5.9 | 12.4 | 18.8 | 26.0 | 32.7 | 46.3 | 52.4 | 58.7 | 62.0 | 64.0 |
| n2 | 0.0 | 29.8 | 42.4 | 49.7 | 50.4 | 48.7 | 44.5 | 39.2 | 36.2 | 28.3 |
| n3 | 0.0 | 0.3 | 0.3 | 2.9 | 3.7 | 4.5 | 5.7 | 6.5 | 19.7 | 22.5 |
| n4 | 27.4 | 24.3 | 20.7 | 17.3 | 14.1 | 11.4 | 8.8 | 6.3 | 4.3 | 4.4 |
| n5 | -0.0 | 10.7 | 10.5 | 12.7 | 15.3 | 17.2 | 19.6 | 20.7 | 21.1 | 21.3 |
| n6 | 8.7 | 12.4 | 22.6 | 34.2 | 44.7 | 50.0 | 53.6 | 55.2 | 59.9 | 61.1 |
| n7 | 9.3 | 24.4 | 33.2 | 40.8 | 47.0 | 54.2 | 59.3 | 63.2 | 67.0 | 66.8 |
| n8 | 8.6 | 12.9 | 18.8 | 25.6 | 33.7 | 39.7 | 47.5 | 59.6 | 61.6 | 65.2 |
| n9 | 0.0 | 1.7 | 3.6 | 4.6 | 5.4 | 6.3 | 6.8 | 6.6 | 5.4 | 1.1 |
| n10 | 0.0 | 9.0 | 11.3 | 15.5 | 15.8 | 17.2 | 32.2 | 33.5 | 37.5 | 42.1 |
| n11 | 0.0 | 11.2 | 18.4 | 17.2 | 19.4 | 32.1 | 29.7 | 28.8 | 35.0 | 37.4 |
| n12 | 0.0 | 8.0 | 9.3 | 16.3 | 19.0 | 19.1 | 23.2 | 33.0 | 40.4 | 43.7 |
| n13 | -0.0 | 0.9 | 2.4 | 6.5 | 11.4 | 12.1 | 13.9 | 17.7 | 21.9 | 25.0 |
| n14 | 0.1 | 3.3 | 3.6 | 7.0 | 13.2 | 19.2 | 24.4 | 27.4 | 29.4 | 38.5 |
| n15 | -0.0 | 0.2 | 0.7 | 0.6 | 0.4 | 2.0 | 4.5 | 19.1 | 28.5 | 31.0 |

Zero- and mean-ablation give the same ordering (tables in `results/phaseA_tables.md`); zero is harsher on the
clocks, mean is milder on n2 late (19.5 vs 28.3 pp at t=10).

**Case drops at t=10 (resample, pp)** — x9max / x9second / x9below / blind:
n0 35/48/43/10 · n1 60/68/64/12 · n2 28/20/29/10 · n3 21/13/24/5 · **n4 36/7/0/1** · n5 20/12/23/4 ·
n6 57/70/61/10 · n7 66/71/66/10 · n8 57/72/65/12 · n9 0/1/1/1 · n10 37/34/44/6 · n11 25/33/40/9 ·
n12 39/42/45/9 · n13 24/12/27/9 · n14 31/32/40/5 · n15 32/17/33/6.

**Groups, final-state-only ablation (accuracy in x9max / x9second / x9below)**

| ablated | mean | resample |
|---|---|---|
| none | .934 / .952 / .957 | |
| comparators n1,6,7,8 | .840 / **.000** / .089 | .478 / .067 / .168 |
| clocks n0,10,11,12,14,15 | .125 / .433 / .231 | .158 / .181 / .195 |
| record n4 | .876 / .821 / .962 | **.570** / .887 / .955 |
| n5,n9 | .807 / .928 / .792 | .740 / .834 / .730 |
| n3,n13 | .715 / .890 / .678 | .628 / .777 / .577 |
| n2 | .692 / .823 / .759 | .653 / .749 / .664 |

The mean-ablation rows for comparators and clocks reproduce FINDINGS.md's table (.840/.000/.085 and
.127/.437/.229) with independent code and data.
All-step resample of any single neuron leaves 13–64% accuracy (n9: 64%, n3: 53%, n15: 48%, others 13–37%).

**Modes of the most common final gate pattern 64975 (29.0% of final states; acc 0.958 inside it).**
Resample of one mode's coordinates at h_10 (accuracy inside the pattern; agreement similar):

| |lambda| | period | main neurons | energy share | acc after resample h_10 | after resample h_9 |
|---|---|---|---|---|---|
| 1.010 | – (integrator) | n12, n0, n14, n11 | .455 | .905 | .780 |
| 0.973 | – (slow) | n0, n12, n11, n14 | .115 | .842 | .728 |
| 0.951 | **8.42 (clock)** | n12, n10, n14, n11 | .313 | .864 | .809 |
| 0.937 | **4.32** | n15, n13, n3 | .063 | **.491** | .532 |
| 0.928 | 2 (Nyquist) | n13, n3, n15 | .007 | .824 | .836 |
| 0.837 | 2.89 | n3, n15 | .044 | .553 | .604 |
| 0.735 | 8.37 (damped) | n14, n15, n3 | .042 | **.390** | .419 |
| 0.231 | – | n14, n8 | .006 | .807 | .935 |
| ~0 | – (input-only) | n7, n8, n6, n1 | .289 | .838 | .925 |

Patterns 63951 (8.6%) and 62927 (6.0%) behave similarly (every mode resample costs 15–70 pp). In patterns
65007 and 65519 the eigenbasis is badly conditioned: oblique mode projections have energy shares 3–120x the
state energy, so their modal ablations are not interpretable (kept in `results/phaseA_modes.json`).

### Phase A, on-manifold check: conditional resampling (DRI request; `python3 phaseA_cond.py`, N = 400k, seed 241061, ~6 CPU-min)
The donor is another input from the same stratum, so the patched state is a value the neuron really takes in that
context. Strata: **S0** none (as above); **S1** same gate pattern of h_t + same x9 case + same age of the max;
**S2** same (age1, age2) cell + same quintile bins of m1 and m2; **S3** = S1 and S2. Singleton strata keep
their own value (<= 1.1% of rows). Accuracy drop in pp at t=10 (t=5 in `results/phaseA_cond.json`):

| stratum | n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 42.5 | 63.9 | 28.2 | 22.6 | 4.4 | 21.5 | 61.2 | 67.1 | 65.2 | 1.0 | 42.0 | 37.4 | 43.6 | 24.9 | 38.5 | 31.0 |
| S1 | 30.5 | 50.0 | 27.2 | 18.5 | 0.3 | 9.5 | 48.1 | 54.0 | 51.0 | 0.2 | 27.7 | 24.2 | 35.1 | 23.5 | 30.7 | 26.3 |
| S2 | 16.2 | 47.9 | 5.7 | 4.9 | 0.2 | 4.7 | 46.3 | 52.1 | 48.6 | 0.0 | 13.2 | 9.2 | 16.0 | 6.9 | 15.5 | 9.0 |
| S3 | 15.3 | 46.9 | 5.4 | 4.3 | 0.1 | 4.2 | 45.1 | 50.9 | 47.4 | 0.1 | 12.6 | 8.7 | 14.5 | 5.9 | 14.5 | 8.6 |

Modes of pattern 64975, donors inside the pattern (accuracy after resampling h_10's mode coordinates; pattern
accuracy .958):

| stratum | integrator | slow | 8.42 clock | **4.32** | Nyquist | 2.89 | **8.37 damped** | 0.231 | input-only |
|---|---|---|---|---|---|---|---|---|---|
| S0 | .904 | .842 | .864 | .491 | .823 | .552 | .385 | .807 | .836 |
| S1 | .907 | .850 | .876 | .516 | .830 | .560 | .432 | .817 | .868 |
| S3 | .951 | .932 | .936 | .857 | .943 | .890 | .841 | .865 | .913 |

Reading:
- Off-manifold damage is not the main story. With S1 donors, which share the gate pattern, x9 case and age of
  the max, the effects stay large for every neuron except n4, n5 and n9. The 4.32 and 8.37 modes still drop
  accuracy to .52 and .43.
- What these neurons and modes carry is mostly position and magnitude information beyond the age of the max:
  the age of the 2nd largest plus the sizes of m1 and m2. Matching on those as well (S2/S3) shrinks n2 from
  27 to 5 pp, n3/n13/n15 from 18–26 to 4–9 pp, the clocks from 24–35 to 9–15 pp, and the 4.32 mode to .86.
- The comparators keep 45–51 pp under S3. Their value depends on x9's exact size relative to M1 and M2, which no
  stratum matches.
- n4's final-step effect disappears once the x9 case is matched: its last-step role is the x9-is-the-max flag.

**Flag for the ledger (H-M3; mode part withdrawn after review, see Review corrections (d)):** H-M3 withdraws n3/n13 as causal "harmonics". Here, on-manifold (S1) resampling
of n3 or n13 alone at t=10 costs 18.5 and 23.5 pp. Resampling the period-4.32 mode that loads mostly on n15, n13
and n3 costs 44 pp in the dominant pattern (.958 → .516). So n3/n13 and the 4.32 mode **are causally important**;
the withdrawal should not be read as "not causal". These checks do not test the specific *harmonic* reading,
i.e. that they resolve higher spatial frequencies. They show the content is mostly (age2, m1, m2) information.
The E29 pair-ablation ranking (n3/n13 near the least damaging pairs) is not contradicted: it used mean-ablation of
a pair under different controls.

### Phase B: encoding (pooled fits; train 1M, test 200k)
Reproduce: `python3 phaseB.py 1000000 200000 && python3 reportB.py > results/phaseB_tables.md` (about 25
CPU-min). The hinge families come from `phaseC2.py` (below); `results/phaseC_tables.md` has the combined
table. Held-out R^2 at t = 10 (complete per-step tables in `results/phaseB_tables.md` and `results/phaseC_tables.md`):

| family (size grows downward) | n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L: x_t, m1..m3 | .268 | .884 | .857 | .148 | .973 | -.015 | .918 | .967 | .857 | .084 | .212 | .224 | .250 | .424 | .374 | .177 | .321 |
| C: + clock phasors | .680 | .949 | .924 | .238 | .981 | .142 | .972 | .980 | .918 | .178 | .835 | .734 | .682 | .487 | .730 | .556 | .732 |
| A1: cell age1 | .873 | .990 | .986 | .314 | .998 | .248 | .993 | .998 | .987 | .322 | .825 | .796 | .816 | .582 | .877 | .659 | .849 |
| A12: cell (age1, age2) | .971 | .998 | .996 | .732 | .999 | .461 | .999 | .999 | .998 | .480 | .942 | .933 | .950 | .814 | .960 | .879 | .955 |
| A123K | .993 | .999 | .998 | .912 | 1.000 | .675 | .999 | 1.000 | .999 | .683 | .983 | .983 | .989 | .930 | .989 | .963 | .989 |
| H1: A123K + comparator hinges, lag 1 | .994 | .999 | .999 | .923 | .999 | **.998** | .999 | .999 | .999 | **1.000** | .985 | .987 | .990 | .936 | .990 | .967 | .996 |
| Hall: A123K + hinges, all lags | .997 | .996 | .997 | **.985** | .996 | .999 | .997 | .996 | .997 | 1.000 | .996 | .996 | .997 | **.988** | .997 | **.994** | .997 |
| A12K-Hall | .994 | 1.000 | 1.000 | .969 | 1.000 | .999 | 1.000 | 1.000 | 1.000 | 1.000 | .997 | .997 | .997 | .978 | .997 | .988 | .998 |

At t <= 3 every neuron is fit exactly (R^2 = 1.0000), because the cell then fixes the full order of the prefix.
The smallest family reaching R^2 >= 0.99 at t=10: A1 for n1, n4, n6, n7; A12 for n2, n8; A123 for n0; H1 for n5,
n9, n12, n14; H3 for n10, n11; Hall for n15; **none** for n3 (.985) and n13 (.988), which only clear 0.99 up to t = 6–9.

Compact formulas at t=10 (pooled least squares, 400k train / 100k held out; `results/headline_formulas.txt`;
M1, M2 = top two of x0..x8):
- comparator n7: pre = −13.17 x9 + 10.93 M1 + 1.97 M2 (R^2 .986); n1, n6, n8 have the same form with
  (−10.6, 7.8, 2.4), (−11.0, 7.7, 2.7), (−12.3, 7.8, 4.5), R^2 .87–.93. Gains on M1 and M2 differ by neuron, so
  the four rectified comparators place x9 against different mixes of M1 and M2.
- record detector n4: pre = 10.16 x9 − 8.52 M1 − 1.37 M2 (R^2 .992): an opposite-sign comparator, on when x9 is a
  record.
- **n9 is a delayed comparator**: pre_t = −5.00 n7_{t−1} − 1.08 n8_{t−1} − 0.36 n1_{t−1} − ... (R^2 .9999). n5 is a
  weaker one (adding comparator hinges takes R^2 from .675 to .998). Both are on when the PREVIOUS input was close to
  the running max (the comparators were near zero one step earlier).
- clocks: m1 times a phasor of age1, plus m2 times a phasor of age2, explains .62–.81 (n0, n10–n12, n14); cells on
  (age1, age2) reach .93–.97; ages of the top 3 reach .98–.99.
- Fast neurons n3, n13, n15 need the comparator history at every lag (Hall) to reach .985–.994: they hold how far
  each recent input fell below the running max, information that ranks and ages alone don't carry.

Per gate pattern (gates of h_9, t=10; `results/phaseB_tables.md`): fitting within a pattern helps the small
families (L: .51 within vs .45 pooled in 64975; .64 vs .04 in 65519) but not A12/A123K (within and pooled agree to
.005). Plain linear in raw x0..x9 within a pattern reaches only .07–.72, so the final gate pattern alone does not fix
the linear map: the earlier gates matter.

Mode basis (final pattern 64975): the integrator and slow mode are linear in (x_t, m1..m3) (R^2 .99/.94); the
**8.42 clock mode is m1 x U(age1)** (A1: .962/.951 for Re/Im; C phasor family .82/.85), quantifying H-M1; the
4.32, 2.89, 8.37 and Nyquist modes need (age1, age2, age3) (.88–.95; A1234K up to .97; hinge families not tried here);
the input-only mode is exactly linear (1.000).

### Phase C: replacement test (v2; N = 1M train / 200k val / 200k eval)
Reproduce: `python3 phaseC2.py 1000000 200000 200000 && python3 reportC.py > results/phaseC_tables.md` (about 30
CPU-min, peak ~1.5 GB); `python3 phaseC_diag.py` (about 8 CPU-min). v1 without hinges: `python3 phaseC.py`.
Network on this evaluation set: accuracy 0.95247.

| replaced (cumulative) | agreement | accuracy | agree: x9max | x9second | x9below | blind |
|---|---|---|---|---|---|---|
| n4 | .9979 | .9519 | .994 | .993 | .999 | .996 |
| + n2 | .9935 | .9523 | .989 | .989 | .994 | .981 |
| + n5, n9 | .9793 | .9471 | .975 | .979 | .980 | .951 |
| + comparators n1, 6, 7, 8 | .9608 | .9481 | .970 | .968 | .959 | .714 |
| + n3, n13 | .9555 | .9479 | .957 | .958 | .955 | .611 |
| + clocks = **all 16** | **.9554** | **.9536** | .953 | .960 | .955 | .542 |
| all 16, without hinge families (v2 eval set) | .9425 | .9505 | .925 | .941 | .945 | .302 |

Single-neuron replacement (others real): agreement .975–.998 (n4 .998, n9 .997, n2 .995; n12 .975, n0 .976,
n10 .976, others .978–.983). Single groups: n4 .998, n2 .995, n5/n9 .981, n3/n13 .976, comparators .967, clocks .963.
Adding neurons one at a time, best first, agreement falls steadily from .998 to .957 after 10 neurons and then
stays at .953–.955 (`results/phaseC_tables.md`).
All-16 accuracy minus network accuracy = 2.06% (replacement fixes a network error) − 1.95% (introduces one) =
+0.11 pp (paired s.e. about 0.045 pp).

**How precise must a formula be?** Add i.i.d. Gaussian noise to the TRUE pre_10 of every neuron so that each has
R^2 = 1 − eps:

| 1 − R^2 | 1e-2 | 3e-3 | 1e-3 | 3e-4 | 1e-4 | 3e-5 | 1e-5 |
|---|---|---|---|---|---|---|---|
| agreement | .612 | .770 | .875 | .938 | .967 | .982 | **.990** |

99% agreement from unstructured per-neuron errors needs R^2 ≈ 0.99999 for every neuron. Our formulas (1 − R^2
1e-4 to 1.5e-2 at t=10, median 3e-3) do much better than noise of the same size (95.5% rather than ~75%), because their errors are
correlated in directions the readout ignores. But neuron R^2 is a poor guide to behavioural fidelity.
**Is the limit the variables or the neuron split?** Fitting the 10 logits directly from the same variables
(Hall family at t=10) gives agreement .9664 (A12K-Hall .9593): only about 1 pp better than going neuron by neuron.
So the variable set, not the decomposition into neurons, is what limits agreement.

### Phase D: where the all-16 replacement disagrees (v2, 200k eval; `python3 phaseD.py`, results/phaseD.json)
- Disagreement 4.46%. It falls on **52.9% of the network's errors** but only 2.0% of its correct answers: the
  formulas reproduce the network's correct behaviour well and its errors poorly.
- **Near ties and the blind region**: inputs with global m2 − m3 < 0.05, or m1 − m2 < 0.05, or blind make up
  19.6% of the mass and **78.5% of disagreements** (rate 17.9% vs 1.19% elsewhere). By m2 − m3 bin
  [0, .01), [.01, .02), [.02, .05), [.05, .1), [.1, .2), [.2, .5), [.5, inf): disagreement .307, .231, .138, .068,
  .033, .015, .012. The network's error rate in each bin is about the same (.434, .272, .143, .062, .030, .014, .011).
- Blind region (1.1% of the mass): 45.8% disagreement, 11.4% of all disagreements.
- Far from ties (80.4% of the mass): disagreement 0.80% on network-correct inputs but 48% on its (rare, 0.83%)
  errors there.
- Rare gate patterns: inputs whose full gate pattern occurs once in the 200k disagree 5.4% of the time, against
  0.8% for patterns seen >= 100 times. Within the far-from-ties set the trend remains (1.42% → 0.13%), and so does the
  network's own error rate (1.08% → 0.09%).
- When they disagree, both pick mostly the 2nd or 3rd largest (replacement: rank 2 46%, rank 3 31%, max 11%).
- Link to T239 [inferred]: the residual is the network's error set. It is concentrated near ties, in the blind region
  and in rare gate patterns, and its rate tracks the network's own error rate bin by bin. That fits the owner's
  hypothesis that "the error part is the unstructured part". It does not prove it: richer variables might still
  capture it.

### Final per-neuron table
Causal effect = accuracy drop (pp) when h_10[j] is resampled (Phase A; step of the peak effect in brackets).
R^2 = held-out at t=10 for the validation-chosen family. Replacement = agreement when only that neuron is replaced
at every step (v2). Confidence refers to the role label.

| neuron | role | causal effect | encoding formula (t=10) | held-out R^2 | replacement agreement | confidence |
|---|---|---|---|---|---|---|
| n0 | clock (slow/integrator modes) | 42.5 (t10) | A123: (age1..3)-cell linear in m1..m5; ~ m1·U(age1) + m2·V(age2) (.66) | .9975 | .976 | medium |
| n1 | x9 comparator | 64.0 (t10) | ReLU(−10.6 x9 + 7.8 M1 + 2.4 M2) (.90); A1 cell (.999) | .9991 | .979 | high |
| n2 | running max | 28.3 (peak 50 at t5) | ≈ 12.9 m1 + 3.8 m2 + ... (.86); A12 (.9997) | .9997 | .995 | high |
| n3 | fast mode (4.32/2.89/Nyquist), late-step | 22.5 (t10) | needs the comparator history at every lag (Hall) | .9854 | .978 | low |
| n4 | record detector (x9 > M1) | 4.4 (peak 27 at t1) | ReLU(10.16 x9 − 8.52 M1 − 1.37 M2) (.992); A1 (.9996) | .9996 | .998 | high |
| n5 | delayed comparator (weak) | 21.3 (t10) | linear in lag-1 comparators + m1, m2 (.946); H1 (.999) | .9992 | .983 | medium |
| n6 | x9 comparator | 61.1 (t10) | ReLU(−11.0 x9 + 7.7 M1 + 2.7 M2) (.93); A1 (.9993) | .9993 | .983 | high |
| n7 | x9 comparator (main) | 66.8 (t10) | ReLU(−13.17 x9 + 10.93 M1 + 1.97 M2) (.986); A1 (.9995) | .9995 | .982 | high |
| n8 | x9 comparator (M2-weighted) | 65.2 (t10) | ReLU(−12.3 x9 + 7.8 M1 + 4.5 M2) (.87); A12 (.9989) | .9989 | .981 | high |
| n9 | delayed comparator | 1.1 (peak 6.8 at t7) | ReLU(−5.00 n7_{t−1} − 1.08 n8_{t−1} − ...) (.9999) | .9999 | .997 | high (role); nearly inert causally |
| n10 | clock (8.42 mode) | 42.1 (t10) | (age1..3, K)-cell + hinges lags 1–3 (H3) | .9969 | .976 | medium |
| n11 | clock (8.42 mode) | 37.4 (t10) | H3 | .9968 | .983 | medium |
| n12 | clock (integrator + 8.42 mode) | 43.7 (t10) | H1 | .9974 | .975 | medium |
| n13 | fast mode (4.32/Nyquist) | 25.0 (t10) | Hall | .9876 | .980 | low |
| n14 | clock + damped 8.37 mode | 38.5 (t10) | H1 | .9974 | .979 | medium |
| n15 | fast mode (4.32), late-step | 31.0 (t10) | Hall | .9943 | .978 | low |

## Review corrections (independent review, branch claude/t218-tie-tail-labels, `review/neuron_account/README.md`)
An independent replication (own code, Philox RNG, same sizes) confirms the core findings: single-neuron
replacement costs, held-out R^2, D1, the mode table and the noise curve all reproduce to within 0.003. It found no
reuse of inputs between train, validation and eval. Corrections, adopted here:

- **(a) The all-16 number depends on the implementation.** Agreement is .9554 here and .9659 in the review
  (.9425 vs .9602 without hinges). With 200k eval rows the s.e. is ~0.04 pp, so this is an implementation effect
  (pooled fallback, ridge in sparse cells, float32 order), not sampling. Read C1 as "about 95.5–96.6%, target 99%
  missed". **The accuracy match is not robust**: +0.11 pp here, +0.82 pp in the review. The "within 0.1 pp" near-miss
  should not be quoted. The robust statement is that the formula model is at least as accurate as the network
  (consistent with D2).
- **(b) Framing.** With all 16 neurons replaced, every state is a function of x, so the all-16 number tests **only
  the t=10 formulas**. `best` all-16 = `best_final_step_only` = .955395. Earlier-step formulas enter only the
  partial replacements. **Recurrent variant, done** (`python3 phaseC3.py`, ~20 CPU-min, `results/phaseC3.json`):
  set the whole state h_{t0} = ReLU(f_{t0}(x)) from the formulas, then run the TRUE recurrence for t0+1..10.

  | t0 | ≤2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|
  | agreement | 1.0000 | .9995 | .9985 | .9966 | .9939 | **.9900** | .9832 | .9778 | .9554 |
  | accuracy | .9525 | .9524 | .9526 | .9529 | .9535 | .9547 | .9562 | .9570 | .9536 |
  | relative state error | <1e-6 | .0025 | .0049 | .0097 | .0129 | .0152 | .0173 | .0201 | .0287 |
  | agreement, blind | 1.000 | .991 | .984 | .967 | .938 | .905 | .841 | .784 | .542 |

  [measured] The formula states are a ≥ 99% faithful substitute for the network's internal state up to t0 = 7.
  The pre-registered agreement level holds through step 7 and fails over the last 2–3 steps, where state errors
  of 1.7–2.9% reach the near-tie readout. The blind region degrades first.
- **(c) n9 is trivial; n5 is overstated.** n9's fitted coefficients are just its row of W_hh (−5.03 n7, −1.09 n8,
  −0.36 n1, ...). "n9 is a delayed comparator" is correct but can be read off the weights; it is not a fitted
  discovery. For n5, with exact weights, the comparator columns explain R^2 .67 at t=2 but only .36 at t=10, and
  least squares on the true lag-1 comparators plus x_t gives .73. The .998 of H1 needs the A123K cells and base as
  well. So n5 is **only partly** a delayed comparator: they are a large minority of its input, not its definition.
  B2 is corrected accordingly.
- **(d) Modes.** The **most damaging mode is the damped period-8.37 mode** (.39), not the 4.32 mode (.49), and the
  same holds at h_9. Mode damage is **not specific**: resampling a random 1-D subspace of the active coordinates
  gives mean accuracy .45, and a random 2-D subspace .30 (the modes are 1–2-D). Mode resampling therefore mostly
  measures the readout's sensitivity to any incoherent perturbation (A5). Per unit energy the 8.37 mode stands out
  somewhat; the 4.32 mode is at random-direction level.
  - Consequence for the H-M3 flag above: its mode part is **withdrawn**. The conditional-resampling evidence
    (n3 / n13 cost 18.5 / 23.5 pp under S1) says n3/n13 are causally used, but every neuron except n4/n5/n9 costs
    24–54 pp under S1. So they are used, not specially important. H-M3's withdrawal as a *specific* causal harmonic
    stands.

## Compute accounting
Single-threaded, `ulimit -v 2500000`, at most 2 processes at once. Compute = exact float64 RNN forward passes
(16x16 recurrence, about 600 FLOPs per step) plus least-squares cell fits (bincount normal equations).
- Phase A: 480 single-step ablations x 200k inputs x (10−t) steps, plus 66 group/all-step runs: about 1.5e8
  sequence forwards (~1e11 FLOPs), ~5 CPU-min. Modes: ~1 CPU-min.
- Phase B: 1.2M forwards; 8 families x 10 steps of cell fits (largest 1M x 11 design): ~25 CPU-min.
- Phase C: v1 ~20 CPU-min; v2 (12 families x 10 steps, 1M train, plus 3 x (16 + 6 + 6 + 16) replacement runs on
  200k): ~30 CPU-min; diagnostics ~8 CPU-min. Phase D: ~1 CPU-min. Exploratory checks (n5/n9 hinges, lags,
  headline formulas): ~10 CPU-min.
- Total: about 1.7 CPU-hours. Input generation: ~5M Gaussian sequences overall.
- Precision: the network runs in float64. The chosen formula values are stored as float32 (relative error 1e-7,
  far below the fit errors of 1e-2 to 1e-3).

## Claims
- A1 [measured] No neuron is redundant in the late steps except n9: resample-ablating any other single neuron at
  t=10 costs 4–67 pp (n4: 4.4, others >= 21 pp); n9 costs <= 6.8 pp at every step.
- A2 [measured] Roles are time-local: n4 matters most at t=1 (27 pp) and fades (4 pp at t=10); n2 peaks at
  t=4–5 (50 pp); comparators and clocks rise with t; n3 and n15 matter only from t=8–9 on; n5 is ~10–21 pp
  from t=2 onward (so not only "after a near-record x8").
- A3 [measured] Case selectivity holds for the comparators (mean final ablation: x9second .000 vs x9max .840)
  and for n4 at the final step (x9max loses 36 pp, others <= 7), not for the clocks, n2, n3/n13, n5 (all cases hit).
- A4 [measured; Review (d): 8.37 most damaging, and not specific vs random subspaces] In the dominant final pattern, the period-4.32 and damped period-8.37 modes (n3/n13/n14/n15)
  are the most damaging modes to resample (to .49 and .39) although they carry 6% and 4% of state energy; the
  8.42 clock and the integrator cost 5–9 pp.
- A6 [measured] On-manifold (S1: same gate pattern, x9 case and age of max) resampling keeps most effects: comparators 48–54 pp, clocks 24–35, n3/n13/n15 18–26 pp; the 4.32/8.37 modes .52/.43. Matching age2 and m1/m2 bins too (S3) removes most of them, except the comparators' (45–51 pp). This contradicts reading H-M3's withdrawal as "n3/n13 not causal".
- A5 [inferred, weakened by A6] The single-neuron damage is mostly an off-manifold effect of incoherent states plus the
  readout's large-cancelling weights (FINDINGS §3), so ablation size measures sensitivity, not "amount of role".
- B1 [measured] Comparators n1/6/7/8, n4 and n2 are encoded at held-out R^2 >= .999 by cells on the age of the max
  (A1) or of the top 2 (A12), linear in (x_t, m1..m3). n7 ≈ ReLU(−13.17 x9 + 10.93 M1 + 1.97 M2) and
  n4 ≈ ReLU(10.16 x9 − 8.52 M1 − 1.37 M2) alone reach .986/.992.
- B2 [measured; corrected in Review (c)] n9 is a delayed comparator (its W_hh row); n5 only partly. Order statistics give R^2 .68 at t=10; adding the previous
  step's comparator values gives .998/.9999 (n9 ≈ −5.00 n7_{t−1} − 1.08 n8_{t−1} + small terms). This replaces the
  FINDINGS label "fire after near-record x8" with a general mechanism that runs at every step.
- B3 [measured] Clocks need cells on the ages of the top 2–3 (R^2 .93–.99). Phasors of the top-3 ages (family C) explain .68–.84.
  The 8.42 mode coordinate itself is m1 x U(age1) at R^2 .96 (pattern 64975).
- B4 [measured] n3, n13 and n15 are the least compressible neurons (best R^2 .985/.988/.994 at t=10). They need the
  rectified comparator history at every lag.
- C1 [measured; Review (a),(b): implementation-dependent .955–.966, t=10 formulas only; recurrent injection keeps >= .99 up to t0 = 7] **Pre-registered target missed**: all 16 replaced gives agreement .9554 (target .99) and accuracy
  .9536 vs .9525 (+0.11 +- 0.05 pp; target within 0.1 pp). Without hinge variables: .9425.
- C2 [measured] Per-neuron R^2 is a poor proxy for behaviour. Each formula has R^2 >= .985, yet replacing a single
  neuron costs 0.2–2.5 pp agreement. Unstructured noise needs 1 − R^2 ≈ 1e-5 on every neuron for 99% agreement.
- C3 [measured] A direct fit of the logits from the same variables reaches only .966, so the variable set limits the
  account, not the neuron decomposition.
- D1 [measured] Replacement disagreements cover 53% of the network's errors and 2.0% of its correct answers. 78.5%
  lie in the 19.6% of mass near a tie (gap < .05) or in the blind region. The disagreement rate tracks the network's
  error rate across gap bins and gate-pattern rarity.
- D2 [inferred] What the account misses is mostly the network's own error set. The formulas describe a "clean"
  2nd-argmax algorithm (accuracy 95.4%, slightly above the network) and do not reproduce which near-tie inputs the
  network gets wrong. This supports the T239 owner hypothesis but does not establish it.

## Alternatives not ruled out
- Richer interpretable variables (hinges of other neurons, clock-gate history, products of gaps with ages)
  might push agreement above 96.6%. Only comparator hinges were tried. The limit in C3 holds for these families only.
- Family choice by validation R^2 optimises the wrong objective (pre-activation MSE, not margin). Fitting to
  logit-margin loss, or nonlinear (e.g. ReLU-aware) fits, were not tried.
- The fitted cells are lookup tables (up to ~2,900 cells x 44 coefficients per neuron and step at t=10). They are
  interpretable in their variables but not compact: description length was not measured (compare H-A1).
- The noise-injection ceiling assumes i.i.d. errors across neurons. Real formula errors are correlated, which is
  why 95.5% is reached at 1 − R^2 ~ 3e-3.
- Conditional resampling only matches the listed strata. Donors still differ in earlier gates and in unmatched values (x9's exact size, m3, the rest of the inputs).
- Oblique mode coordinates depend on the gate pattern; donor states from the same final pattern can still
  have different earlier gates.

## Failure modes
- Mode ablations are only meaningful in well-conditioned patterns (64975, 63951, 62927).
- Replacement fidelity collapses in the blind region (agreement .54). There the network's output is a small
  difference of near-zero states, and relative formula errors are largest.
- Fixed-family replacements fail because early-step errors propagate: with A12 at all steps, replacing n4 alone gives
  .880 agreement (v1, `results/phaseC.json`). Exactness at t <= 3 matters.
- Memory: cell fits at 1M rows need chunked prediction (`CellLinear.predict`). The first v2 attempts ran out of
  memory under the 2.5 GB cap, and the pooled fallback was switched from lstsq to normal equations; this only
  affects cells unseen in training (0–0.22% of eval rows in Phase B).

## Proposed shared-code changes
None. Follow-ups proposed for the board: (1) independent re-check of C1/C2/D1 with different code and seeds;
(2) try margin-loss fits and hinges of the clock gates to see whether >= 99% agreement is reachable with
interpretable variables; (3) measure description length of the fitted cell tables (link to H-A1).
