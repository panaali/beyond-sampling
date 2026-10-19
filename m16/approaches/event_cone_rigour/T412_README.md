# T412: fast rigorous M5t (`m5t_fast.py`)

↑ [event_cone_rigour README](README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

Owner: Claude child session_01333jBMKr2Pn3Hfse5vLAUW. Branch `claude/t412-m5t-fast` (off `claude/t400-bound-rigour`;
T400 merges). New files only; `rigour.py` is not edited.

## Question
Can the per-simplex M5t bound `rigour._bounds_tangent` (arb, ~36 ms per simplex) be replaced by a batched float
computation in which every rounding error is bounded and rounded outward, at >= 100x the speed and with
essentially the same tightness?

## Pre-registration (pushed before any timing or tightness number)
Reference: `rigour._bounds_tangent` in arb (python-flint), as called by `rigour.mass_cell(..., tangent=True,
zbins=24, zratio=23/20)` (the `a_m5.py ... t` settings).

Test set: every simplex that `mass_cell(..., tangent=True)` evaluates on the 27 k = 4 cells (t0 = 2.25) and the
28 k = 4 cells (t0 = 2.5) of `data/bench_182_k4.npz` (cells with >= 30 MC hits, as in `a_m5.py`).

PASS requires all three:
1. **Speed:** >= 100x per simplex against arb `_bounds_tangent` (wall time per simplex, same machine, single core,
   fast version batched over the simplices of a cell).
2. **Tightness:** fast hi <= 1.01 x arb hi on >= 99% of the test simplices.
3. **Soundness:** 0 violations of (a) fast hi >= arb lo and fast lo <= arb hi on every test simplex, and (b)
   end-to-end `mass_cell` with the fast bounds gives ub >= MC - 4 s.e. on every cell (MC and hits from
   `results_a_m5t_*.json`; s.e. = MC / sqrt(hits)). Also reported, not part of the line: the fraction of
   simplices with fast hi >= arb hi (1 - 1e-12); any exception is explained.

Then (reported, no line): time the 10-D region 3 (`data/regions_1-8-2.npz` index 3, event=True, t0 = 1.5) with
the fast bounds against the arb reference (374 s, ub 6.19e-11).

Soundness comes from the error analysis below; the tests only check it.
