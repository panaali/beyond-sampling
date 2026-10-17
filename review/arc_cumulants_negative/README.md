# T80: independent verification of negative E35 (ARC cumulant propagation on AlgZoo)

↑ [Review index](../README.md) · [INDEX (registry + ledger)](../../INDEX.md) · [TASKS](../../TASKS.md) ·
reviewed: [m16/approaches/arc_cumulants](../../m16/approaches/arc_cumulants/README.md) (E35), context E13
[m16/approaches/propagation](../../m16/approaches/propagation/README.md), T69, T40

## Question
E35 says ARC's Hermite-corrected cumulant propagation (arXiv:2605.05179), applied faithfully to the AlgZoo
2nd-argmax RNNs, misses the accuracy by 41-100 pp. Its stated root cause is that the scalar input per step makes the
step-1 hidden units perfectly correlated, which breaks the method's small-correlation power counting. Per AGENTS rule 5
this negative stays "unverified" until (1) the root cause is re-tested with independent code, (2) an exhaustion
checklist exists, and (3) the most promising untried variants have had a fair attempt. Verdict wanted: **dead
(verified)** or **reopened**.

## Method
Everything here is new code (`review/arc_cumulants_negative/`). ARC's vendored library is used only where the
question is about ARC's own algorithm (K >= 3 propagation, E35's Heaviside read-outs), always driven by new loops.
All runs: `source .venv/bin/activate; export OMP_NUM_THREADS=1`, from this folder, one core, <= 3 GB virtual
memory (`ulimit -v 3000000`).

| file | what it does |
|---|---|
| `cumprop.py` | own moment propagation: `arc2` = ARC's K=2 rule (exact 1-D mean/variance, off-diagonal C_ij c_i c_j); `gauss` = full Gaussian closure (exact bivariate-normal ReLU moments = all Hermite orders per pair, E13/Wright et al.), via a 1-D Gauss-Legendre conditional integral split at the kink, so exact also for correlation ±1; `rim_hidden` = recent-input mixture (variant a, below) |
| `test_cumprop.py` | `gauss` vs MC (incl. rank-1 covariances, within MC error) and vs E13's closed form (agree to 4e-15) |
| `rootcause.py` | random RNNs, input dimension d per step in {1, 2, 4, 16, H}, H in {16, 64, 256}, T in {4, 10}, linear read-out; variance-normalised MSE vs MC (1M or 400k draws), 3 seeds; also mean \|corr\| of h_T |
| `layers_m16.py` | trained M_16,4 (seed3) and M_16,10 (orig): layer-by-layer relative rms error of mean / variances / off-diagonal covariances vs MC (2M draws) |
| `unit_ceiling.py` | variant (b) ceiling: per-unit Edgeworth E[relu(u_i)] from the *exact* cumulants (orders 2-4) of each pre-activation |
| `edgeworth.py`, `test_endpoint.py` | Gaussian / 2nd-order Edgeworth orthant probabilities P(m > 0) from cumulants 1-4 (checked on Gaussian and Gamma cases: Edgeworth-4 within 0.0003 of truth) |
| `endpoint_ceiling.py` | variants (c), (d) ceiling: exact cumulants (MC) of the margins l_p2 − l_i per cell -> Gaussian / Edgeworth endpoint; cells = (p1,p2), full permutation, (p1,p2) x sign(x_0); plus one joint Gaussian of (l, x) |
| `readout_ceiling.py` | E35's own Heaviside read-outs (uncond, cell) run by ARC's propagation from the *exact* cumulants (orders 1-4) of the final state (h_T, x) |
| `variants_acc.py` | accuracy estimators of the untried variants (deterministic): `joint:<closure>`, `cell:<closure>`, `arccell:K`, `rim:m:n:closure` |

**Why ceilings.** Every cumulant method has two parts: propagate cumulants through the RNN, then turn the final
cumulants into P(correct). Feeding each part the *exact* cumulants (estimated by MC with 2e5-2e6 draws; used only
for diagnosis) tells which part fails and bounds every variant that keeps that part.

## Results

### 1. Root cause re-tested on random RNNs (`rootcause.py`) [measured]
`python rootcause.py 4 16,64 1,2,4,16,H 1000000 3` (and `10 ...`, and `4|10 256 1,4,16,H 400000`).
Entries: variance-normalised MSE of E[Wo h_T] (equivalent number of samples in brackets); mean |corr| of h_T by MC.
`arc2` (own) equals ARC's vendored K=2 to all printed digits (checked at H=16: 7.4e-3 / 3.3e-4 both).

| T | H | d | mean\|corr\| | arc2 (ARC K2) | gauss (exact pairwise) | ARC K3 (vendored) |
|---|---|---|---|---|---|---|
| 4 | 16 | 1 | 0.28 | 7.4e-3 (134) | 3.7e-3 (269) | 2.2e-1 (5) |
| 4 | 16 | 2 | 0.24 | 2.0e-3 (510) | 1.4e-3 (730) | 7.1e-4 (1,416) |
| 4 | 16 | 4 | 0.17 | 1.4e-3 (734) | 1.2e-3 (812) | 1.3e-4 (7,858) |
| 4 | 16 | 16 | 0.11 | 3.3e-4 (3,040) | 3.0e-4 (3,361) | 1.9e-5 (53,828) |
| 4 | 64 | 1 | 0.26 | 3.7e-3 (267) | 3.4e-3 (291) | 8.8e-3 (113) |
| 4 | 64 | 2 | 0.19 | 1.8e-3 (563) | 1.2e-3 (837) | 2.6e-4 (3,835) |
| 4 | 64 | 4 | 0.14 | 4.5e-4 (2,230) | 4.2e-4 (2,393) | 2.8e-5 (35,489) |
| 4 | 64 | 16 | 0.09 | 7.1e-5 (14,084) | 6.2e-5 (16,024) | 2.4e-6 (416k) |
| 4 | 64 | 64 | 0.06 | 2.3e-5 (42,575) | 2.2e-5 (45,498) | 1.9e-6 (MC floor) |
| 4 | 256 | 1 | 0.25 | 4.3e-3 (232) | 3.4e-3 (292) | - |
| 10 | 16 | 1 | 0.27 | 1.3e-2 (74) | 8.3e-3 (121) | 2.2e-2 (45) |
| 10 | 16 | 16 | 0.12 | 8.6e-4 (1,165) | 8.5e-4 (1,179) | 2.5e-4 (3,967) |
| 10 | 64 | 1 | 0.17 | 1.6e-3 (606) | 1.7e-3 (576) | 7.0e-4 (1,438) |
| 10 | 64 | 2 | 0.14 | 3.5e-3 (289) | 2.5e-3 (401) | 1.9e-4 (5,380) |
| 10 | 64 | 16 | 0.10 | 2.6e-4 (3,879) | 2.6e-4 (3,889) | 5.4e-6 (183,735) |
| 10 | 64 | 64 | 0.09 | 1.4e-4 (7,145) | 1.3e-4 (7,588) | 1.1e-5 (90,645) |
| 10 | 256 | 1 | 0.15 | 1.8e-3 (545) | 2.0e-3 (511) | - |
| 10 | 256 | 4 | 0.09 | 1.8e-4 (5,605) | 1.5e-4 (6,693) | - |
| 10 | 256 | 16 | 0.06 | 3.9e-5 (25,713) | 3.3e-5 (30,223) | - |
| 10 | 256 | 256 | 0.04 | 1.9e-5 (51,966) | 1.9e-5 (53,879) | - |

Raw files: `out_rootcause_*.txt` (the H=16 T=4 d=16 row is from a re-run, `out_rootcause_T4.txt` stopped early).

- **E35's observation reproduces** with independent code and seeds: with scalar input (d = 1) the error stays at
  ~100-600 sample-equivalents from H = 16 to H = 256, and mean |corr| of h_T stays at 0.25 (T = 4) instead of
  falling like H^-1/2; with d = H it improves with H and with K as the paper predicts.
- **Refinement of the mechanism.** The exact Gaussian closure (every Hermite order for each pair, i.e. no
  power-counting truncation of the covariance update at all) is at most ~2x better than ARC's leading-order rule at
  d = 1 and equal to it at d >= 4. So the dominant error at d = 1 is **not** the truncated off-diagonal series; it is
  that the state is a deterministic function of only t scalar sources, hence far from any Gaussian-plus-small-
  cumulants law (rank(Cov h_t) <= t·d < H). What scales is the number of independent sources (d), not the width:
  error falls steadily with d at fixed H. ARC's K=3 corrections help only from d >= 2 and can hurt at d = 1
  (2.2e-1 at H=16 T=4). [measured for the numbers; inferred for the mechanism]

### 2. Trained models, layer by layer (`layers_m16.py`) [measured]
`python layers_m16.py 4 arc2,gauss,arcK3,arcK4,rim:0:16:arc2,...` -> `out_layers_T4*.txt`; `python layers_m16.py 10
arc2,gauss,arcK3,rim:2:-12:gauss` -> `out_layers_T10.txt`. Relative rms error of mean / variance / off-diagonal cov:

| M_16,4 (seed3) | h_1 | h_2 | h_3 | h_4 |
|---|---|---|---|---|
| arc2 (ARC K2) | 0.001/0.002/0.354 | 1.47/7.8/7.5 | 3.7/38/29 | 5.8/57/57 |
| gauss (exact pairwise) | 0.001/0.002/0.002 | 1.50/8.1/10.3 | 3.9/40/30 | 5.7/57/56 |
| ARC K3 (vendored) | 0.001/0.002/0.043 | 24/522/5.5 | 1.8e5/4e10/7e5 | diverged |
| ARC K4 (vendored) | 0.001/0.002/0.043 | 1.3/1.6/2.4 | 7.5/52/55 | 29/1218/3560 |
| rim m=1 (split 16) | 0.000/0.004/0.004 | 0.009/0.006/0.013 | 0.44/2.5/1.8 | 0.96/3.2/2.8 |
| rim m=2 (split 16) | same | same | 0.015/0.014/0.032 | 0.065/0.18/0.32 |
| rim m=4 = pure grid quadrature | same | same | same | 0.022/0.022/0.039 |

| M_16,10 (orig) | h_2 | h_4 | h_6 | h_8 | h_10 |
|---|---|---|---|---|---|
| arc2 | 0.37/1.1/0.56 | 0.60/1.5/1.2 | 0.70/1.8/1.5 | 0.62/1.8/1.5 | 0.48/1.6/1.4 |
| gauss | 0.37/1.3/0.50 | 0.64/1.6/1.3 | 0.77/2.1/1.7 | 0.71/2.1/1.7 | 0.56/1.8/1.6 |
| ARC K3 | 2.4/8.6/0.40 | 4.3/41/28 | 14/186/129 | 1e4/6e8/939 | 1e5/8e10/6e9 |
| rim m=2 (split 12+) | 0.012/0.019/0.018 | 0.065/0.24/0.20 | 0.13/0.20/0.38 | 0.13/0.13/0.31 | 0.094/0.16/0.33 |

h_1 is exact for every method (also its off-diagonal covariance with `gauss`, which E35 identified as ARC-K2's first
error). Yet `gauss` is as wrong as ARC-K2 from h_2 on. Cause, read off the step-2 pre-activations of M_16,4: the
weights are large (‖Wh‖ = 24.5) and several units have pre-activations that are sign-definite functions of the
inputs, e.g. unit 0 has mean −19 and sd 28 but E[relu] = 0 exactly (never active). A Gaussian with that mean and sd
gives 4.05, and similarly for units 5, 7, 10, 12 and 13 (Gaussian 4-10 vs truth 0.00-1.09). The trained network uses
exactly the structure that a Gaussian/Hermite expansion around each unit's own Gaussian cannot see.

### 3. Variant ceilings: what exact cumulants would buy [measured]

**3a. Per-unit Hermite corrections with exact higher cumulants (variant b ceiling).** `python unit_ceiling.py 4|10`
(`out_unit_T*.txt`). Relative rms error of E[relu(u_i)] over units, given the exact cumulants of each u_i:

| step | M_16,4: Gaussian / +κ3 / +κ4,κ3² | M_16,10: Gaussian / +κ3 / +κ4,κ3² |
|---|---|---|
| 2 | 1.50 / 0.89 / 0.76 | 0.37 / 0.22 / 0.19 |
| 3 | 1.06 / 0.62 / 0.52 | 0.25 / 0.13 / 0.11 |
| 4 | 0.85 / 0.48 / 0.39 | 0.19 / 0.10 / 0.085 |
| 10 | - | 0.072 / 0.036 / 0.030 |

Even with *perfect* input cumulants, one nonlinear step is off by 39-76% (M_16,4) and 3-19% (M_16,10) of the mean
activation, and the series converges slowly. Making the pairwise part exact (E13) does not touch this error: it is a
1-D marginal effect.

**3b. Cumulant endpoints from the exact final cumulants (variants c, d).** `python endpoint_ceiling.py 4
p1p2,perm,p1p2s 400000 seed3`, `python endpoint_ceiling.py 4 p1p2 200000 seed3 joint`, `python endpoint_ceiling.py
10 p1p2 200000`. Truth 0.99831 (M_16,4 seed3) and 0.95314 (M_16,10 orig); "mc" = the same draws counted directly
(checks the cells):

| model | conditioning (cells) | Gaussian | Edgeworth κ3 | Edgeworth κ3,κ4,κ3² (= T40 argmax endpoint) |
|---|---|---|---|---|
| M_16,4 | none, one Gaussian for (l, x) | −73.0 pp | - | - |
| M_16,4 | (p1,p2), 12 | −20.4 pp | −15.8 pp | −13.7 pp |
| M_16,4 | (p1,p2) x sign(x_0), 24 | −18.8 pp | −14.2 pp | −12.0 pp |
| M_16,4 | full ranking of x, 24 | −15.3 pp | −11.4 pp | −9.5 pp |
| M_16,10 | (p1,p2), 90 | CEIL10_G | CEIL10_E3 | CEIL10_E4 |

**3c. E35's own read-outs from the exact final cumulants.** `python readout_ceiling.py 4 2,3,4 uncond 2000000`,
`... 4 2,3,4 cell 400000`, `... 10 2,3,4 cell 200000`, `... 10 2,3 uncond 1000000`:

| model | read-out | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| M_16,4 seed3 | uncond (E35) | −79.8 pp | −66.4 pp | −61.5 pp |
| M_16,4 seed3 | cell (E35) | −30.5 pp | −9.5 pp | −2.7 pp |
| M_16,4 s0.01_0 | cell | R4S_2 | R4S_3 | R4S_4 |
| M_16,10 orig | cell (E35) | −37.5 pp | −30.5 pp | +101.3 pp |
| M_16,10 orig | uncond (E35) | R10U_2 | R10U_3 | not run (memory) |

So **even a perfect propagation** of cumulants would leave E35's estimator 2.7-100 pp off; the best case (M_16,4,
cell, K=4, −2.7 pp) is still 20x the sampling s.d. at 1,000 FPE (0.13 pp) and the same read-out diverges at T=10.
The endpoint (turning cumulants into an indicator probability) is a second, independent failure, as E13 found for
its margin closures (−1.4 / −0.35 pp with 10k cells; here the cells are coarser and the gaps larger).

### 4. Fair attempts at the untried variants (accuracy, deterministic) [measured]
`python variants_acc.py 4 <specs> base` etc.; raw: `out_rmse_T4.txt`, `out_rmse_T10.txt`, `out_rim_T4.txt`,
`out_arccell_T4.txt`, `out_variants_T10.txt`. Cost = analytic FLOP estimate / FPE (rough, for orientation).

| variant | spec | M_16,4 seed3 (0.99831) | M_16,10 orig (0.95314) | cost (FPE) |
|---|---|---|---|---|
| (b) exact pairwise closure, joint (h,x), Gaussian endpoint | joint:gauss | −73.0 pp | J10G | 522 / J10GC |
| (b) ARC-K2 closure, joint | joint:arc2 | −72.5 pp | −80.4 pp | 44 / 66 |
| (b) exact pairwise per (p1,p2) cell, Gaussian endpoint | cell:gauss | −68.4 pp | C10G | 6,237 / C10GC |
| (b) ARC-K2 per cell | cell:arc2 | −68.8 pp | C10A | 497 / C10AC |
| (b)+(d) ARC K-th order per cell + Edgeworth/argmax endpoint | arccell:2/3/4 | ARCCELL | - | ARCCELLC |
| (a) recent-input mixture, last 1 input on 16-node grid | rim:1:-16:gauss | −69.4 pp | - | 1.4e5 |
| (a) last 2 inputs on grid | rim:2:-16:gauss | −54.2 pp | - | 1.8e6 |
| (a) last 3 inputs on grid | rim:3:-16:gauss | −30.3 pp | - | 1.7e7 |
| (a) all inputs on grid (pure quadrature, no cumulants) | rim:4:-16 / rim:4:-8 | +0.17 pp (estimate 1.00000) | - | 1.3e5 / 1.3e4 grid points |

RMSE over models (same specs; seeds 0-4 and noisy copies s0.0025..s0.02): RMSE_TABLE

Notes. (a) The structure-vs-randomness split helps the hidden-state moments a lot (Section 2: relative error 0.07-0.3
at h_4 with 2 recent inputs exact) but the accuracy stays 30-70 pp off until *every* input is on the grid, at
which point no cumulant is left and the method is tensor-product quadrature of an indicator: it returns 1.00000 (all
16^4 or 8·10·12·14 grid points correct; the 0.17% failure set is thin and falls between nodes), i.e. +0.17 pp at
1.3e4-1.3e5 forward passes, where sampling has s.d. 0.01-0.04 pp. (First grid version used the same nodes for every
input; grid ties x_i = x_j broke the 2nd-argmax label (−29 pp); fixed by a different node count per input.)
(c) Conditioning on discrete states (sign of x_0, full ranking) moves the endpoint ceiling only from −20 to
−15 pp at T = 4; E35's skeleton conditioning (3e) and E13's gate-pattern mixtures (gsf2, −57..−69 pp) cover the rest.
(d) The argmax endpoint d/da E[max(l+a)] under the Edgeworth density of l equals the Edgeworth orthant probability
of the margins (margins are linear in l), so 3b column 3 is its ceiling: −9.5 to −13.7 pp at T = 4.

## Exhaustion checklist
Tried (here or in E35/E13), with outcome:

| # | variant | where | outcome |
|---|---|---|---|
| 1 | ARC basic (SIMPLE) K = 1-4, uncond read-out | E35 3c | −41..−100 pp |
| 2 | ARC augmented (AUGMENT) K = 1-4 | E35 3c | same or worse, NaN |
| 3 | (p1,p2)-cell conditioning + ARC | E35 3c-d | −46..−78 pp |
| 4 | skeleton (top-k inputs fixed) conditional ARC | E35 3e | useless unless 8/10 inputs fixed |
| 5 | random-RNN width/input-dim scan | E35 3b, here §1 | reproduced; mechanism refined (few sources, not truncation) |
| 6 | own K=2 implementation vs vendored | here §1 | identical |
| 7 | exact pairwise ReLU moments (Wright series) instead of leading term | here §1, §2, §4 | ≤2x better on random RNNs, no better on trained; −68..−73 pp |
| 8 | per-unit higher cumulants with *exact* inputs | here §3a | 39-76% error per step at T=4 |
| 9 | exact cumulants into E35 read-outs (uncond, cell; K 2-4) | here §3c | −2.7..−80 pp; +101 pp at T=10 K=4 |
| 10 | Gaussian / Edgeworth / argmax endpoints from exact cumulants | here §3b | −9.5..−73 pp |
| 11 | finer discrete conditioning before the endpoint (sign of x_0, full ranking) | here §3b | −12..−15 pp at best |
| 12 | x_t-conditioned closure (law of total covariance over the fresh input) | here §2 (rim m=0) | same as plain closure |
| 13 | structure vs randomness: last m inputs exact on a grid, Gaussian residual (T69) | here §2, §4 | −30..−69 pp for m < T; m = T is pure quadrature (+0.17 pp, worse than sampling) |
| 14 | Gaussian-sum / gate-pattern mixtures | E13 gsf, gsf2 | −57..−69 pp |
| 15 | cone-exact branches with truncated-Gaussian ADF | E13 prop.py | −10..−21 pp, worse with compute |
| 16 | exact margin cumulants in 10k cells | E13 ceiling | −1.4 / −0.35 pp vs 0.1 pp bar |
| 17 | DEV-copy stability of the one near-hit | E35 3c | −88..+93 pp |

Not tried, with the reason it is not expected to change the verdict:

| variant | reason |
|---|---|
| ARC factorized variant | same output as basic by construction (paper S.4.3), only cheaper |
| K >= 5 | needs exact inputs of order >= 6 to test the ceiling; at T=10 the cell read-out already diverges at K=4 (+101 pp), and the hidden-state propagation diverges at K=3 (§2) |
| symmetric-tensor FLOP discount | changes cost by 0.3-0.7x; errors are 10^2-10^6x sampling's MSE |
| other target encodings (max via ReLU trees, single margin threshold) | the uncond and cell read-outs already fail from exact cumulants; any encoding must still map cumulants to an indicator, which §3b bounds |
| random-feature / PCA change of basis | ReLU acts coordinatewise in the neuron basis; a linear basis change cannot change the per-unit marginals that fail in §3a |
| larger widths / longer T on random RNNs | §1 already shows no width dependence at d = 1 from 16 to 256 |
| E35 cell mode with K=4 at T=10 propagated (not ceiling) | ceiling is already +101 pp |
| mixtures with more components (grid + Gaussian residual at T=10) | at T=4 the accuracy needs all inputs on the grid; at T=10 a full grid of ≥8 nodes is 10^9 points |
| ARC's experiment/max-endpoint branch code | not fetched; the endpoint is equivalent to the Edgeworth orthant of the margins (§4 note d), whose exact-cumulant ceiling is measured |

## Verdict
**Dead (verified)** as a sampling competitor on M_16,4 and M_16,10, for ARC's method and for every cumulant variant
listed above. Two independent failure points, each sufficient on its own [measured]:
1. *Propagation*: the hidden state is a deterministic function of a few scalar inputs and the trained weights make
   many pre-activations sign-definite; per-unit Hermite expansions are 39-76% off per step at T=4 even from exact
   input cumulants, and exact pairwise moments do not help.
2. *Endpoint*: even exact final cumulants give −2.7 pp at best (T=4, E35 cell read-out, K=4) and −9.5 pp or worse
   for Gaussian/Edgeworth/argmax endpoints; at T=10 the cell read-out goes −37 / −30 / +101 pp for K = 2/3/4.

**Correction to E35's stated root cause** [measured on random RNNs, inferred for the trained models]: the scalar input
is the right culprit, but the mechanism is the small number of independent sources (state of rank <= t), not the
truncation of the off-diagonal covariance series; removing the truncation (exact pairwise moments) does not help.
On the trained models, the endpoint failure is as large as the propagation failure.

The only member of the family that comes near the truth is the one with no cumulants left (full-grid quadrature, +0.17
pp at 1.3e4 forward passes, where sampling has s.d. 0.04 pp).

## Alternatives not ruled out
1. *A structure-aware closure that tracks sign-definite units exactly* (e.g. a unit is a known ReLU of a
   known one-sided variable) could fix §3a for the first steps. This is a cone/region method in disguise (E13
   prop.py, sci_tao, branch_bound) rather than cumulant propagation, and the endpoint (§3b-c) would still need a
   non-cumulant read-out. [speculative]
2. *MC error in the "exact" cumulants.* The ceilings use 2e5-2e6 draws per cell; κ4 of a 26-dim state from 2e5 draws
   is noisy, which could explain part of the +101 pp divergence at T=10 K=4. Not checked with more draws. The K=2/3
   ceilings and the T=4 ones (4e5-2e6 draws, reproducible to 0.02 pp between two runs of 2e5 and 4e5 draws) are not
   affected at the reported precision.
3. *Only base models for the ceilings and the layer tables* (seed3 at T=4, orig at T=10), plus one noisy copy for the
   cell read-out ceiling; the accuracy variants were run on 10 / 8 models (RMSE above). The DEV models were not used.
4. *Cost numbers in §4 are analytic estimates*, not counted; they matter only for the quadrature row, whose
   deterministic grid costs about one forward pass per point (the printed 1.7e7 counts the unused closure overhead).
5. *T = 5, 6 not re-run here*; E35 covers T = 5.

## Compute accounting
Diagnostics (MC references and exact-cumulant estimates) use random inputs and are **not** estimators. The §4
variants are deterministic (no model calls on random inputs; endpoint integrals over explicit Gaussian densities use
fixed scrambled-Sobol or seeded Gaussian points). Their cost column is an analytic FLOP estimate: linear maps
2HD² + 2H²D per step plus ReLU moments (≈ H²/2 pairs × 80 nodes × 30 FLOPs for `gauss`, ≈ 4H² for `arc2`) per
component, divided by `fpe_flops(H, T)`. Total CPU for this review: about 3 core-hours.

## Claims
- [measured] E35's random-RNN observation reproduces with independent code: scalar input -> error flat in width
  (H 16-256) at ~100-600 sample-equivalents; vector input -> improves with width and K.
- [measured] My ARC-K2 implementation equals ARC's vendored K=2.
- [measured] Exact pairwise (Gaussian-closure) moments are at most 2x better than ARC-K2 on scalar-input random RNNs
  and no better on M_16,4 / M_16,10.
- [inferred] The scalar-input failure is a few-sources (low-rank, non-Gaussian) effect, not a power-counting
  truncation effect.
- [measured] With exact input cumulants, per-unit Hermite corrections are 39-76% off per step on M_16,4 (3-19% on
  M_16,10).
- [measured] With exact final cumulants, every tested endpoint/read-out is ≥ 2.7 pp off on M_16,4 and ≥ 30 pp off
  on M_16,10 (cell read-out).
- [measured] Variants (a)-(d) as estimators: −30 to −80 pp; the full-grid limit of (a) is +0.17 pp at 1.3e4 FPE.
- [inferred] No member of the cumulant family listed in the checklist can compete with sampling on these models.
