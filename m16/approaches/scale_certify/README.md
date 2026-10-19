# T426: how certified pruning (T398b machinery) scales with H and T

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · scale-study registry: branch `claude/t421-scale-study`,
`m16/approaches/scale_study/` · machinery: branch `claude/t373-event-cone`, `m16/approaches/event_cone/` (T390/T398/T398b),
rigour tools: branch `claude/t400-bound-rigour` (T400/T412/T415). Owner: Claude child session_018MUzmq5iomeZWKGnvmq7Zv.

## Question
[OWNER_QA 6.7](../../../OWNER_QA.md#q6-7): on M_16,10 at t0 = 2.5, an early triple (1,8,2) is pruned to k = 10 with
~1.5e4 survivors and every certificate verifies, but late triples cannot be pruned and all sampled event hits lie on
(9,3,2). How do survivor counts, growth per step, and the feasibility of a per-triple certificate change as the
2nd-argmax RNN grows in H (16, 24, 32, later 48, 64) and T (10, later 15, 20)?

## Pre-registration (written 2026-10-19 before any bnb run on a grid model and before reading any grid-model MC output)
Already run before this was written (validation only): `mc.py orig 2.5 2e6` on ARC's M_16,10 (7 hits, all (9,3,2),
P = 1.41e-10 +- .53e-10, vs T400's 1.55e-10 +- .37e-10, all on (9,3,2)); `bnb_run.py orig 2.5 1,8,2 --kmax 3`
(survivors 2 / 18 / 39, identical to T398b). MC on the three grid models was launched but not read.

**Models.** Per grid point (H, T), the registry model `m_H_T_cpu0` (primary panel, seed 0; weights checked out from
claude/t421-scale-study into `data/t426/weights/`). Reference: ARC's M_16,10 (`orig`), from T398b's published numbers.

**Event E1(t0)** = {x_p1 >= x_p2 >= x_p3 + t0, x_p3 >= every low, argmax L = p3} (T340's E1), x ~ N(0, I_T).
- **Threshold rule (model-free):** t0 = 2.5 at T = 10; at other T, the t0 whose slab mass P(g23 >= t0) equals T = 10's
  at 2.5 (4.03e-5): t0 = 2.2604 (T = 15), 2.1301 (T = 20) (`mc.t0_rule`). DELTA = t0/7.5 (normalisation loss
  2T Phi(-7.5) <= 1.3e-12). The rule does not target P(E1); the measured P(E1) is reported as it comes
  (expected ~1e-10 at H = 16, lower at larger H since accuracy rises).

**Step 1, dominant triple (plain MC in the slab, `mc.py`).** Exact conditional sampler of x given g23 >= t0 (triple
uniform; x_p3 by inverse CDF of its 1-D law on a 4e5-point grid; lows and highs by truncated-normal inverse CDF).
P(E1) = P(slab) hits / n with Clopper-Pearson bounds. n = 2e7 (seed 0); if < 5 hits, add 2e8 (seed 1).
Dominant triple = most hits (tie: smaller second-smallest position b).

**Step 2, branch and bound (`bnb_run.py`, `core.py` = event_cone bnb.py + cone_count.split_fast + relax2.prune_stage
copied line for line with H, T read from the model).** T398b setting: relax2 stages 0-3, split depth 6, DELTA =
t0/7.5, EPS 1e-7. Per point, up to 3 triples:
- A = the dominant triple;
- B = (1,8,2) (fixed comparator, early: b = 2);
- C = the second most-hit triple if it has >= 2 hits, else the late triple (T-1, T-3, T-2).
  If A = B, C as stated; if A is unknown (0 hits), A is skipped.

CPU cap: **2 CPU-h per (model, triple) run**, one process, `ulimit -v 2500000`, memory guard 6e4 survivors.
A step stopped by the cap is reported as censored (children made / survivors so far).

**Reported per run:** survivors S_k per complete k, growth g_k = S_k / S_(k-1), the projection
S_T^proj = S_K g_K^(T-K) from the last complete step K, and the projected bnb CPU to k = T (sec per child at step K
times projected children, children/survivor ratio of step K held fixed).

**Soundness check (could fail):** every MC hit of the triple must lie in a surviving cell at every complete k
(`hits_lost` = 0). A hit inside a pruned cell refutes the pruner on that model.

**Pre-registered line, per-triple certificate feasible:** S_T^proj <= 1e5 AND projected bnb CPU <= 200 CPU-h.
**The machinery "holds" at a grid point** iff the dominant triple A is feasible by this line (it fails on orig: the
event mass sits on (9,3,2); T398b's (9,3,2) run is the reference if published). "Partly holds": only B or C feasible.

**Step 3, mass charge (if affordable).** For one early triple per point that completes k = T with <= 2e4 survivors,
charge the survivors with T412's mass_cell_fast plus T415's geom_fast (H-C1 scope); labelled **provisional** while
geom_fast is under audit (T420).

**Predictions (written now; scored, not tuned):** [speculative] (P1) at fixed k, S_k of (1,8,2) rises with H (more
neurons split per step), >= 3x from H 16 to H 32 at the deepest common k; (P2) the dominant triple stays "late-ish"
(b >= 3) and is not certifiable at any grid point; (P3) per-step growth rises with T at fixed H (more remaining steps
make the relaxation looser).

## Method
_(filled in after the runs)_

## Status
running

## Results
_(pending)_

## Compute accounting
_(pending)_

## Claims
_(pending)_

## Alternatives not ruled out
_(pending)_

## Failure modes
_(pending)_
