# T424: do the rare-event estimators (H-R1 K3, graded-g12) keep holding as H and T grow?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · owner request: [OWNER_QA 6.7](../../../OWNER_QA.md#q6-7) · models: [T421 scale study](https://github.com/panaali/clean-neurons/tree/claude/t421-scale-study/m16/approaches/scale_study)

Owner: Claude child session_01VFFq5SDaq6V34Nkq4k9kAj (branch `claude/t424-scale-rare`). DRI: session_01CD3g9Cvc3qk3pM11eyXXzj.

## Question

The exact-walk K3 estimator (H-R1) and the graded-g12 rule (estimator_t366) estimate E1(t0) = {pick = rank-3 position,
g23 > t0} within 2x down to p ~ 1e-10 on M_16,10-class models (except triple-tie-dominated points). Do they keep holding on
2nd-largest RNNs with more neurons H and more steps T (the T421 grid), and where do they stop?

## Pre-registration (committed before any estimate or truth on a T421 model)

Nothing in this section changes after the first scored run except by a dated, flagged amendment below it.

**Estimators (frozen; DEDUCTIVE, no random inputs).**
- Source: claude/t218-tie-tail-labels @5b4a1250 r1_cont/ and claude/t286-cheap-frontier @620eb47b; all 7 sha256 of
  `r1_cont/t366_freeze_sha256.txt` asserted ([freeze.py](freeze.py)).
- **K3** = T364's frozen H-R1 estimator: T340c lines (3 exact-law Gauss nodes in g12 | v2), the T354 walk fix and segment
  start, T_hi = 6 (`t354.est_seg(model, 3, t0s, t2s=(), tmax=6)`).
- **graded** = estimator_t366 (`t366.est_e1`, T_hi = 6): 10 Gauss-Legendre nodes on g12 in [0, .4] + 3 law nodes beyond.
- Generalisation to any (H, T) in [gen.py](gen.py): the same functions with the (16, 10) constants made parameters (T
  logits, T(T-1)(T-2) triples, the T-4 lows, the v2 density T(T-1) phi, the Phi^(T-2) profile of g23 | v2, v4 = max of
  T-3; FLOP charges read H from the model); rows walked in chunks (memory). No other change.
- **Gate (before any scoring):** at (16, 10), gen.py must equal the frozen code to relative 1e-9 on every t0 for both
  estimators on 2 registry models ([validate.py](validate.py); frozen graded run in the T388 row-chunking, which equals
  the unchunked run to 2e-16). If the gate fails, nothing is scored until it passes.

**Grid and panel.** T421 registry points, in this priority order as they land: (16,10), (24,10), (32,10), (48,10),
(64,10), (16,15), (16,20), (32,15), (32,20), then the rest. **Primary panel: cpu seeds 0, 1, 2 of each point** (fixed
before seeing any accuracy beyond the registry), all included whatever their accuracy (flagged if < .90). At (16,10) the registry so far has only cpu seed 0 (plus
zoo 0-4): there the primary panel is cpu 0-2 if cpu 1-2 land before the scoring of that point, else **cpu 0 + zoo 0 +
zoo 1** (the anchor; disclosed as a stream mix). Supplementary
if compute allows: cpu seeds 3-5, and the zoo panel at (16,10) / (32,10); reported separately, not in the verdict.

**Thresholds (rule fixed now, no estimate or truth used).** [thresholds.py](thresholds.py):
predicted p(t0) = P_T(g23 > t0) x c x (1 - acc), with P_T the exact model-free gap law for T inputs, acc the registry's
plain-MC accuracy, c = 0.015 (fixed from T366's (16,10) conditional rates). Per model two thresholds: the t0 on a .05
grid with predicted p closest to **1e-7** and to **1e-8**. Disclosure: the rule uses one model-level common quantity
(1 - acc), no rare-event quantity; T366 showed a fixed factor can miss the range by ~10x, so out-of-range truths are
expected sometimes and are scored, not dropped.

**Truths.** Own conditional-law MC ([truth.py](truth.py), generalising T371's truth.py to any T): condition on v3, draw
the two tops above v3 + t0 and the T-3 lows below v3, random positions; p = P_T(g23 > t0) x hit rate. Draws in chunks
of 2e5 until >= 100 hits (cap 4e9 draws). A point is scored only with **>= 50 hits**; otherwise reported as "truth
infeasible". Cross-check of truth.py before scoring: at a common threshold (t0 = 1.0, (16,10)) against plain
unconditional MC on the same model (must agree within 3 s.e.).
**Order:** estimates for a grid point are computed and committed (sealed) before its truths are run.

**Scoring per grid point and estimator.**
- r = est / truth; "within 2x" = .5 <= r <= 2.
- **Holds** at a grid point = **>= 80% of its scored primary points within 2x** (with 6 points: >= 5/6).
- "Stops holding in H" = the smallest H at T = 10 (and in T, the smallest T at H = 16; also H = 32 if run) whose
  verdict is fail, given that all smaller points of that axis hold; a non-monotone pattern is reported as is.
- Also per point: the triple-tie share (truth hits with g12 < .1), C (FPE as executed, 1 FPE = 5948 FLOPs x
  flops(H,T)/flops(16,10), flops = T(2H^2+2H) + 2HT), p C, and **F vs plain MC at equal FPE** (Directive 8 amendment 2:
  F_i = e_i^2 C_i / (p_i (1-p_i)), e = est - truth; panel F = mean(e^2) C_eff / mean(p(1-p)), C_eff = max(mean C, 1k)),
  judged against **both bars, strict F <= 1 and lenient F <= 10**.
- Secondary: the same verdict on in-range points only (truth in [1e-9, 1e-6]); walk truncation (rows reaching the
  frozen 5000-cell limit) per call.

**Expectations (written before running) [speculative].** At T = 10, growing H keeps the input-space geometry (same
lines, same 720 triples), so both should keep holding unless wider nets put more error mass in the triple-tie sliver or
in second-order structure the closure (single v4, mean lows) misses. Growing T multiplies the lines (T(T-1)(T-2)) and
the lows (T-4, all at one mean value): the closure is more likely to break with T than with H; cost grows ~T^3-T^4.

**Compute.** CPU only, <= 2 heavy processes, each under `ulimit -v 2500000`. If a point's estimator calls are too slow
to finish, the stop and a compute estimate go to docs/COMPUTE_FRONTIER.md (standing rule), not a silent drop.

## Method

*(filled after the runs)*

## Status

running

## Results

*(filled after the runs)*

## Compute accounting

*(filled after the runs)*

## Claims

*(filled after the runs)*

## Alternatives not ruled out

*(filled after the runs)*

## Failure modes

*(filled after the runs)*

### Gate result (2026-10-19, before any scoring)
`validate.py m_16_10_zoo0` and `m_16_10_cpu0` ([results/validate.json](results/validate.json)): gen.py equals the frozen
code **bitwise** (max relative difference 0.0) at t0 = 1, 1.5, 2, 2.5 for both K3 and graded. C as charged: graded equal;
K3 +0.03% (row chunks lose a little prefix-tree sharing in the first cell). Gate passed.

### Amendment 1 (2026-10-19, after the first T=10 truths; flagged)
Truth stopping: a point with **0 hits after 1e9 draws** stops there (upper bound p < 3e-9 x P_T(g23 > t0), far below
1e-9; such a point is "truth infeasible" and unscored under the original 4e9 cap too, so no verdict can change). Reason:
points with p ~ 1e-12 or less were taking ~3 CPU-h each and blocking the T = 15 / 20 estimates. Points with >= 1 hit
keep the 4e9 cap. Working checkpoints are written to $T424_TRUTH_DIR while running and copied to results/truth when done.
