# t348: gate-pattern census of M_16,10 (T348 + T346 extension)

**Question.** How fast do the probability-relevant gate patterns (activation regions) of M_16,10 grow beyond k = 6,
for all inputs, near-tie inputs and error inputs; and do error patterns grow slowly enough (<= ~3x per step) for exact
sub-cone enumeration of rare events (T346)?

**Method.** Diagnostic sampling, not an estimator. `census.py 1e9 8` on a Colab A100 (t192f2): 1e9 float64 inputs
x ~ N(0, I_10) (one CUDA Philox stream, seed 3480000, regenerated for every pass), gate word w_t = 16 bits of h_t > 0;
pattern = (w_1..w_k) (prefix) or (w_(11-k)..w_10) (suffix), 64-bit order-dependent hash (exact key for k <= 4); distinct
counts by GPU sort + unique in 8 hash partitions. Reported per (k, set): distinct D (lower bound on the number of
regions), singletons f1, Good-Turing coverage 1 - f1/n (sample mass in patterns seen more than once; 1 - coverage
estimates the unseen mass), plug-in exp(H), and N(delta) = fewest patterns covering 1 - delta of the sample.

**Status.** done (2026-10-18 16:40 UTC). Unverified.

## Results [measured]
Validation against the exact k = 4, 5 enumeration (`../results/prefix/k*_regionmass.npz`): exp(H) 302 / 1,546 (exact
302 / 1,546); N(1e-1, 1e-2, 1e-3) at k = 4: 396 / 1,737 / 4,016 (exact 396 / 1,737 / 4,018); k = 5: 2,548 / 14,656
(exact 2,548 / 14,658). Collision check: k = 10 distinct under two independent 64-bit mixers 64,892,589 both.

**Prefix gate patterns (steps 1..k), all inputs (n = 1e9)**

| k | distinct D | growth | exp(H) | coverage 1-f1/n | growth | N(1e-1) | growth | N(1e-2) |
|---|---|---|---|---|---|---|---|---|
| 4 | 13,622 |  | 302 | 1.0000 |  | 396 |  | 1,737 |
| 5 | 140,022 | x10.3 | 1,546 | 1.0000 | x5.1 | 2,548 | x6.4 | 14,656 |
| 6 | 899,923 | x6.4 | 7,152 | 0.9997 | x4.6 | 15,679 | x6.2 | 117,200 |
| 7 | 3,932,888 | x4.4 | 29,577 | 0.9986 | x4.1 | 88,106 | x5.6 | 842,515 |
| 8 | 12,633,752 | x3.2 | 106,587 | 0.9944 | x3.6 | 434,257 | x4.9 | 5,000,799 |
| 9 | 31,722,385 | x2.5 | 341,681 | 0.9836 | x3.2 | 1,924,601 | x4.4 | 21,722,385 |
| 10 | 64,892,589 | x2.0 | 1,015,733 | 0.9620 | x3.0 | 7,729,990 | x4.0 | 54,892,589 |

**Prefix gate patterns (steps 1..k), near-tie inputs, min(g12, g23) < .1 (n = 3.44e8)**

| k | distinct D | growth | exp(H) | coverage 1-f1/n | growth | N(1e-1) | growth | N(1e-2) |
|---|---|---|---|---|---|---|---|---|
| 4 | 12,330 |  | 319 | 1.0000 |  | 421 |  | 1,812 |
| 5 | 117,065 | x9.5 | 1,707 | 0.9999 | x5.4 | 2,838 | x6.7 | 15,983 |
| 6 | 712,578 | x6.1 | 8,325 | 0.9994 | x4.9 | 18,344 | x6.5 | 134,165 |
| 7 | 2,953,547 | x4.1 | 36,972 | 0.9966 | x4.4 | 108,791 | x5.9 | 991,992 |
| 8 | 8,883,849 | x3.0 | 142,155 | 0.9877 | x3.8 | 564,473 | x5.2 | 5,443,590 |
| 9 | 20,615,997 | x2.3 | 467,293 | 0.9667 | x3.3 | 2,542,015 | x4.5 | 17,175,738 |
| 10 | 38,283,669 | x1.9 | 1,322,306 | 0.9304 | x2.8 | 9,161,407 | x3.6 | 34,843,410 |

**Prefix gate patterns (steps 1..k), error inputs, pick != p2 (n = 4.69e7)**

| k | distinct D | growth | exp(H) | coverage 1-f1/n | growth | N(1e-1) | growth | N(1e-2) |
|---|---|---|---|---|---|---|---|---|
| 4 | 10,092 |  | 298 | 1.0000 |  | 423 |  | 1,902 |
| 5 | 79,776 | x7.9 | 1,552 | 0.9996 | x5.2 | 2,771 | x6.6 | 16,937 |
| 6 | 397,588 | x5.0 | 7,634 | 0.9969 | x4.9 | 18,233 | x6.6 | 140,629 |
| 7 | 1,345,487 | x3.4 | 33,436 | 0.9868 | x4.4 | 108,603 | x6.0 | 876,587 |
| 8 | 3,270,834 | x2.4 | 121,340 | 0.9618 | x3.6 | 522,480 | x4.8 | 2,801,934 |
| 9 | 6,284,215 | x1.9 | 352,227 | 0.9161 | x2.9 | 1,972,674 | x3.8 | 5,815,315 |
| 10 | 9,741,755 | x1.6 | 852,223 | 0.8580 | x2.4 | 5,052,754 | x2.6 | 9,272,855 |

**Suffix gate patterns (steps 11-k..10), all inputs**

| k | distinct D | growth | exp(H) | coverage | growth | N(1e-1) | growth | N(1e-2) |
|---|---|---|---|---|---|---|---|---|
| 7 | 29,236,025 |  | 260,872 | 0.9847 |  | 1,506,778 |  | 19,236,025 |
| 8 | 44,548,390 | x1.5 | 510,833 | 0.9751 | x2.0 | 3,393,977 | x2.3 | 34,548,390 |
| 9 | 64,647,779 | x1.5 | 1,007,994 | 0.9621 | x2.0 | 7,668,187 | x2.3 | 54,647,779 |
| 10 | 64,892,589 | x1.0 | 1,015,733 | 0.9620 | x1.0 | 7,729,990 | x1.0 | 54,892,589 |

**Suffix gate patterns (steps 11-k..10), error inputs**

| k | distinct D | growth | exp(H) | coverage | growth | N(1e-1) | growth | N(1e-2) |
|---|---|---|---|---|---|---|---|---|
| 7 | 5,751,346 |  | 290,568 | 0.9241 |  | 1,627,109 |  | 5,282,446 |
| 8 | 7,589,620 | x1.3 | 506,471 | 0.8944 | x1.7 | 2,900,619 | x1.8 | 7,120,720 |
| 9 | 9,718,156 | x1.3 | 848,214 | 0.8584 | x1.7 | 5,029,155 | x1.7 | 9,249,256 |
| 10 | 9,741,755 | x1.0 | 852,223 | 0.8580 | x1.0 | 5,052,754 | x1.0 | 9,272,855 |

## Reading [inferred]
- A figure is resolved only while the unseen mass (1 - coverage) is well below what it measures. D is a lower bound
  everywhere from k = 5 (exact k = 5 has 632,394 regions; the sample sees 140,022; exact k = 6 >= 15.5M vs 0.9M seen):
  **the slowing of D growth is a sampling artefact.** exp(H) is reliable to about k = 8 (all) / k = 7 (errors);
  N(1e-1) while 1 - coverage << 0.1 (all: through k = 10, unseen 3.8%; errors: through k = 8, unseen 3.8%).
- Probability-weighted growth per step is far below the region-count growth (x23-x30 per step, T270): exp(H) grows
  x5.1, x4.6, x4.1, x3.6 (k = 5..8, all inputs) and x5.2, x4.9, x4.4 (k = 5..7, errors); N(1e-1) grows x6.4, x6.2, x5.6,
  x4.9, x4.4, x4.0 (all, k = 5..10) and x6.5, x6.6, x6.0, x4.8 (errors, k = 5..8).
- So error-pattern growth is ~x4-6 per step at the delta = 0.1 / entropy scale, declining with k: between Fable's
  thresholds (<= ~3x revives, 25-44x kills), and resolved only at coarse delta. N(1e-2) and smaller are not resolved at
  1e9 beyond k ~ 7 (errors ~6).

## Alternatives not ruled out
- The decline of the per-step factor at k >= 8 may still be part sampling bias (plug-in entropy and N(delta) are biased
  low as coverage drops). A coverage-adjusted estimator (Chao-Shen; needs the count-of-counts, not saved) or 1e10+
  samples would separate it.
- N(delta) counts patterns, not exact regions: one pattern can be several cells (different start cones); exact growth
  in cells is larger.

## Compute accounting
1e9 inputs x 96 forward passes (12 jobs x 8 partitions) on a shared A100, ~45 min; ~0 marginal Colab units (VM up for
grid3). Two earlier attempts died of CUDA OOM (unpartitioned near-tie set; skewed partitions of exact k <= 4 keys).

## Failure modes
- Exact keys (k <= 4) must be re-mixed before partitioning (skewed top bits); undersampled sets give lower bounds only.
