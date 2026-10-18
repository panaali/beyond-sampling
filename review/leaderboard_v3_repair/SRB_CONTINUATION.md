# Remaining direct srb cost audit: source inventory, not an execution gate

The bounded T204 saved audit does not settle the3.3/1.6 multiplier. The next phase must remain separately claimed and independently source-gated. This note records the concrete source path inspected without importing it:

- `control_variate/estimator.py` loads `alloc.json["plain"]` and builds `cv.make(strat=True, cv=None, alloc=_alloc)`.
- `cv.make` uses c=COST_Q+COST_DRAW=1.6+0.1, chooses n=floor(budget/c), imposes at least four samples per stratum, calls pooled `draw_all`, then Q and a weighted sum. Returned cost is ceil(len(x)*c). The number of rejected/generated candidates returned by `draw_all` is discarded at this call site.
- `rb9.Q` executes nine hidden steps, the last-step offset, breakpoint/target sorting and up to nineteen interval bodies. Each active batch interval recomputes two dense logit products. Interval activity is tested for the whole batch; one active sample retains operations on other rows too. A single-row cost cannot automatically be multiplied into a historical batched run.
- The current harness applies3.3/1.6 to the **whole returned cost**, including the0.1 draw allowance and final ceiling. That is not exactly the same operation as replacing only COST_Q1.6 by3.3. This distinction is arithmetical, not evidence that either interpretation is complete.
- The historical audit already identifies dense recomputation versus the comment's incremental rank-one-update assumption. Current line citations differ; no current exact installed runtime count is available from that comment.

A useful next bounded scope would pin this complete call path and sampler, declare scalar arithmetic/library/input-generation conventions, then validate instrumented-versus-unmodified semantics on fixed analytic controls before any trained run. It must report actual batch shape and interval activity, separate candidate-generation/advice/precomputation, and retain library/control exclusions. Any copied historical multiplier or single-prefix ratio is insufficient. This note authorizes no new execution and claims no new measured factor.
