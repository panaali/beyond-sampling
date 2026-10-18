# T373: exact enumeration inside a rare event's own cone (kill test)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · spec: `review/fable_t346/README.md` section 3.3
(branch `origin/claude/project-handoff-priorities-zpzfm8`, read as a past record only) · T344 section of
`m16/approaches/r1_cont/README.md` (branch `origin/claude/t218-tie-tail-labels`)

## Question
Can the far-gap rare event E(t0) = {the network picks the position of the 3rd-largest input, and g23 = x_(2) - x_(3) >= t0}
be given a **rigorous** probability (exact, or a certified upper bound) by enumerating the exact activation regions
only inside the event's own cone, at a cost near plain sampling? This is the kill test for that idea (Fable T346 idea
E, re-scoped).

## Pre-registration (written and pushed before any test count is run; 2026-10-18)

**Model.** orig = M_16,10 (`alg_zoo.example_2nd_argmax`, weights via `m16/common.py`, float64). t0 = 1.5.

**Cone.** For a position triple (p1, p2, p3) the true event cone is K = {x_p1 >= x_p2 >= x_p3 + t0, x_p3 >= x_l for
every low l} (no proxy: g12 and all low gaps are free).

**A remark made before running (it predicts the result).** The network has no biases, so every activation region
is a polyhedral cone C. C meets K in a set of positive volume iff C meets the homogeneous ordering cone
O = {x_p1 >= x_p2 >= x_p3 >= every low} in positive volume: a generic interior point of C ∩ O has g23 > 0, and
scaling it by a large positive factor puts it in K. So **the exact region count inside K does not depend on t0**,
and K's count equals that of O, which leaves the 7 lows unordered. Consequences:
- I count on O (homogeneous, with T344's [-1,1]^10 box only as LP normalisation). Using the affine K with T344's box
  [-1,1]^10 would be wrong: the box makes g23 >= 1.5 cut cones by a scale artefact. As a check that could fail, I
  also count the affine K with the box [-1e4, 1e4]^10 for k <= 4 on two triples and require identical counts.
- Every region of the full space meets at least one of the 720 ordering cones, so the sum over all 720 triples of
  count_k is >= N_k (T270: N_6 >= 19,740,596). The scaled 10-triple sum should therefore be >= ~2e7 at k = 6 (up to
  sampling noise of the 10 triples), and the GO line N10 <= 2e8 needs an average growth 5->6 below ~1.8x.
  **Prediction: STOP**, with projected N10 ~1e9-1e11.

**Code.** `cone_count_t344.py` is T344's `cone_count.py` copied verbatim (sha256
`0b87cd21828cfc7b9ceb9cfec0efec17cb173bfc49689cd24224c6d185dda801`, from origin/claude/t218-tie-tail-labels).
`cone_count.py` here is that file with only these changes, each listed in its header:
1. `cone()` returns O (or the affine K for the check above);
2. the triple list is the pre-registered random one and the model is orig only;
3. an optional speed-up: an interior point per cell decides which side of each neuron's hyperplane is
   non-empty, so one LP per neuron is solved instead of two. It must reproduce the unmodified algorithm's counts
   exactly for k <= 4 on two triples, otherwise it is not used;
4. the cap is on distinct regions (2e5), checked as in T344 (the step is marked incomplete and counting stops for
   that triple when exceeded). A capped count is a lower bound.

**Triples.** 10 distinct triples from `np.random.default_rng(373)` (`rng.permutation(10)[:3]` repeated, duplicates
skipped): (1,2,0), (9,7,8), (8,9,6), (2,8,7), (5,2,7), (2,8,5), (1,7,8), (1,0,8), (4,1,3), (0,4,2).

**Steps.** k = 1..6 (k = 2..6 reported). Cap 2e5 regions per triple. At most 2 processes, `ulimit -v 2500000`.
Total compute budget 8 CPU-h; if exceeded, the remaining triples are reported as censored at their last complete k.

**Projection.** N10 = 72 x sum over the 10 triples of count_6 x (count_6 / count_5)^4. For a triple capped at
k = 6, its count_6 and growth are lower bounds, so its term is a lower bound.

**GO** iff projected N10 <= 2e8 AND max / median over the 10 per-triple projected terms count_6 x g^4 <= 10.
**STOP** otherwise (scoped negative, with the AGENTS rule-5 exhaustion checklist). If a capped triple makes the
verdict depend on the cap, the verdict is reported as "STOP (lower bound)" only if the lower bound already
exceeds 2e8; otherwise "inconclusive (censored)".

**Event-feasible fraction at k = 6.** The event is decided at step 10, so feasibility at k = 6 is not an LP on the
k = 6 cell. I report a **sampled lower bound**: draw points in K by a proposal (lows iid N(0,1); x_p3 = max(lows) +
Exp(1); x_p2 = x_p3 + 1.5 + Exp(1); x_p1 = x_p2 + Exp(1)), run the network, keep those where it picks p3, and count
the distinct 6-step gate prefixes among them, divided by count_6. The same samples (all of them, not only event
points) validate the enumeration: **every sampled 6-step prefix must be among the enumerated regions** (any miss is
a bug).

## Method
(to be filled in after running)

## Status
Pre-registered; not yet run.
