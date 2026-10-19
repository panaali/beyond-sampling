# T425: locate and repair a rare failure across the T421 scale grid (H-R2, H-R3)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · owner request: [OWNER_QA 6.7](../../../OWNER_QA.md#q6-7), [4.10](../../../OWNER_QA.md#q4-10)

**Pre-registration: committed and pushed before any design run or truth on a grid model (2026-10-19).** Sections marked
*(filled later)* are written after the runs; nothing in the Method changes afterwards except by a dated, flagged
amendment pushed before the runs it governs.

## Question
T402 (H-R3) repaired M_16,10's rare far-gap error E1(2.0) (p 6.8e-9): a 12-weight edit from the line-CMC weight-gradient
of the rare probability plus the accuracy Hessian made it ~17x rarer for -.011 pp accuracy; 0/40 random same-norm edits
did that (T410), and part of the rare error was converted into the common 1st-vs-2nd near-tie error (T402, T411). Does
this keep holding on 2nd-largest RNNs with more neurons H and more steps T (the T421 grid), and how do the cost, the
number of weights needed and the conversion scale?

## Method (pre-registered)

**Models.** T421 registry (branch `claude/t421-scale-study`, `scale_study/registry.json`, weights `weights/m_<H>_<T>_cpu<s>.npz`).
Per grid point, the primary-panel models **cpu seed 0 and cpu seed 1** (fixed by seed number, not by accuracy), seed 1 only
if it has landed in the registry before this session's run queue reaches that point. Grid points are run as they land;
first (16,10), (24,10), (32,10). M_16,10 itself is T402's anchor (not rerun). Every model is reported, whatever its accuracy.

**Event.** E1(t0) = {argmax = position of the 3rd-largest input, g23 = v2 - v3 > t0} (T402's event, any T).
**t0 rule (model-free rule, design estimates only, no truths):** scan t0 = .25, .50, ..., 6.0 upward with the design
line-CMC estimate (2e6 lines, grad off) and stop once it falls below 1e-9; t0 = the scanned grid value whose design P is
nearest 1e-8 in log10 among those in [1e-9, 1e-7] (if none is in range, the nearest to 1e-8 with P > 0, flagged).

**Design (T402's method, generalised to any H, T; `core.py`, `design.py`; sampling-class, conditional, no observed failures).**
1. g_E = dP(E1(t0))/dW by line-CMC: x drawn from its exact law given g23 > t0 (v2 density T(T-1) phi (1-Phi) Phi(v2-t0)^(T-2)),
   v3 integrated exactly along its line over (v4, v2 - t0) with the exact walker, boundary-term gradient; 2e7 lines.
2. For K in a ladder: S_K = the K weights with the largest |g_E| among those with |g_E| > 3 s.e.; accuracy Hessian on S_K by
   paired plain-MC central differences (h = .005; 2e6 inputs for K <= 12, 1e6 for K = 24); direction d = -M^-1 g_E,S with
   M = -Hessian, eigenvalues clipped below at .05 x the largest (T402 amendment R3); unit L2.
3. Step eps = .00025 k, k <= 200 (L2 <= .05): smallest k with design P <= P0/20 by doubling (1, 2, 4, ..., 128, 200) then
   bisection (design P 5e6 lines and design accuracy 1e7 paired inputs, common random numbers for all k); K qualifies if
   the design accuracy change there is >= -.02 pp (T402's 2x/2.5x design margins; T402 note R4's bracket logic built in).
4. Ladder: K = 12 first (T402). If it qualifies, try K = 6, then 3 (stop at the first failure); if not, try K = 24.
   **K\* = the smallest qualifying K** ("weights needed"), edit = its smallest qualifying k. If none qualifies, the
   **fallback** edit (largest K tried, the lowest-design-P grid point meeting the design accuracy limit) is frozen and
   verified anyway, and is reported as a design failure.
5. **Freeze:** `results/<tag>/edit.json` (base and edited weights, sha256) is committed and pushed before verification.

**Verification (`verify.py`; own forward and own conditional sampler, new seeds; run once per frozen edit).**
- E1(t0): exact conditional-law MC (v2 by inverse CDF on its own grid, the T-2 others iid truncated below v2 - t0), base,
  edit and 10 random edits on the same draws. N_E = expected >= 1000 base hits from the design P0 (capped at 3e8); the random
  edits on the first N_R draws (expected >= 300 base hits). Paired ratio and delta-method s.e.
- Random control: 10 edits of the same weights S_K\* with the same L2 norm, Gaussian directions (fixed seed).
- Accuracy: plain MC on N(0, I_T), base and edit on the same 1e8 inputs; random edits on the first 1e7.
- Rare-to-common conversion: on the E1 draws, the picked-rank split at g23 > t0 for base and edit; the change of
  P(pick = largest input, g23 > t0) vs the E1 probability removed; also P(any wrong pick, g23 > t0) = W(t0) (T411).

**"Holds" per model (all three, fixed now).**
- (a) verified E1 ratio <= .1 with ratio + 1.96 s.e. <= .15 (>= 10x rarer);
- (b) accuracy change >= -.05 pp (point estimate, 1e8 paired; gains are not failures);
- (c) beats the random control: no random edit meets ratio <= .1 and accuracy >= -.05 pp, and the targeted ratio is below
  every random edit's ratio.
A grid point holds if every scored model there holds; "stops holding" in H (at T = 10) / in T (at fixed H) = the first point
along that axis that does not hold. Reported per model: t0, P0, K\*, eps, the ratio, accuracy cost, random min / median,
the conversion, design and verification cost (forward-pass equivalents of that model, FLOPs, CPU time).

**Compute.** CPU, one thread per process, <= 2 heavy processes in this container, each under `ulimit -v 2500000`;
helper child sessions (own containers) for large points if needed.

**Predictions [speculative], fixed now.** (1) The method keeps working at T = 10 for H 24-32 (same task, similar circuits),
with fewer than 12 weights not enough at larger H more often; (2) it degrades with T sooner than with H (more steps = more
gate flips along each line and a flatter gradient); (3) the conversion into the common near-tie error persists everywhere
(T411's reading: the common error is training-optimised, the rare one is not).

**Alternatives to keep in mind.** (i) An edit could cut E1 by a global rescaling that any direction achieves: the random
control. (ii) At high accuracy (H >= 24, acc ~.99) E1 at the chosen t0 could be dominated by a different mechanism than
M_16,10's comparator balance: the gradient is computed per model, no mechanism is assumed. (iii) Design estimates could be
biased at edited weights: verification is independent code.

### Amendment A1 (2026-10-19, after a 1/50-size smoke run of the pipeline on m_24_10_cpu0, outputs discarded; before any full design run)
If no grid point of any K tried meets the design accuracy limit, the frozen fallback is the largest K tried at its smallest
k meeting the design P condition (k = 200 if none), status "fallback_noacc"; it is verified and reported like any other edit.
Smoke checks: verify.py on T402's frozen M_16,10 edit (3e7 draws, 4e6 inputs) gives ratio .054 +- .012 (T402 .059), largest-input
picks +9.3% (T402 +9.3%), random min / median .16 / 1.03 (T410 .165 / .80).

### Amendment A2 (2026-10-19 18:5x, after a container restart killed the first runs during the t0 scan; no gradient, edit or truth yet)
Implementation only, for resumability: every stage is memoised on disk (data/t425/ckpt). The 2e7-line gradient is 10 blocks of
2e6 lines, the Hessian accuracy sums 4 blocks, and verification runs in blocks of 1e7 draws / inputs with per-block seeds; N_E and
N_R are rounded up to whole blocks (more draws than pre-registered, never fewer). Rules, sizes and criteria are unchanged.

### Note A3 (2026-10-19 22:4x; implementation only)
The K = 24 Hessian ran out of memory (one 1153-model x 1e5 array); core.acc_plain now streams one model at a time (same
inputs and seeds, identical numbers). The pushes "m_16_10_cpu0 / m_24_10_cpu0 design (no edit)" (68a68282, 37a1ecc4) were
that crash, not design outcomes; run_queue.sh no longer pushes after a crash. Memoised stages are reused.

### Amendment A4 (2026-10-19 23:2x, T421 grid final at 75b6f749; no new design started)
Scope: the 13 kept points (screened (16,15) and (16,20) are dropped; m_16_20_cpu0 is not run). Order: cpu seed 0 at every kept
point first, then seed 1 at each point; queues per container (this one plus helper child sessions):
this: m_24_10_cpu0, m_16_10_cpu0, m_24_10_cpu1, m_16_10_cpu1 | m_32_10_cpu0, m_32_10_cpu1;
helpers: m_24_20_cpu0, m_24_20_cpu1 | m_48_15_cpu0, m_48_15_cpu1; m_48_10_cpu0, m_48_10_cpu1 | m_64_10_cpu0, m_64_10_cpu1;
m_24_15_cpu0, m_24_15_cpu1 | m_32_15_cpu0, m_32_15_cpu1; m_32_20_cpu0, m_32_20_cpu1 | m_64_15_cpu0, m_64_15_cpu1;
m_48_20_cpu0, m_48_20_cpu1 | m_64_20_cpu0, m_64_20_cpu1.
Seed-1 runs not finished when the write-up is due are reported as not run (with their compute estimate in docs/COMPUTE_FRONTIER.md);
a point is then scored on seed 0 alone and flagged. Added report (DRI request): a comparison at matched error rate, (16,10), (24,15),
(24,20), (32,20) (error ~.05-.10) vs the others (.003-.016).

## Status
Pre-registered; running *(filled later)*.

## Results
*(filled later)*

## Compute accounting
*(filled later)*

## Claims
*(filled later)*

## Alternatives not ruled out
*(filled later)*

## Failure modes
*(filled later)*
