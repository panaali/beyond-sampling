# T345: architecture control (feed-forward MLP on the M_16,10 task)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [OWNER_QA 6.1](../../../OWNER_QA.md#q6-1)

Owner: Claude child session_01Na5o6K4TXWJmNtj7N5f76X, branch `claude/t345-ffn-control`.

## Question

Would a network that sees all 10 inputs at once (1-2 ReLU layers, no recurrence) be easier to estimate
mechanistically than ARC's M_16,10 RNN? Concretely: on the same task and parameter budget, (a) how accurate is a
small MLP, (b) how many linear regions does it have compared with the RNN's x23-33 growth per step, (c) can its
accuracy be computed exactly or deductively, and (d) what figure of merit F (vs plain sampling at equal compute) do
our exact-walk tools reach on it, compared with the same tools on M_16,10?

## Pre-registration (written and pushed before any scored run, 2026-10-18)

### Task and models
- Task: alg_zoo `2nd_argmax`, continuous: x ~ N(0, I_10), label = `x.argsort(-1)[..., -2]` (alg_zoo/tasks.py).
- Bias-free ReLU MLPs (bias=False like M_16,10; this keeps the network positively homogeneous, so its linear
  regions are cones through 0, as for the RNN). Sizes (`train.py`):
  - `L1h22`: 10-22-10, 440 parameters; `L1h43`: 10-43-10, 860 (2x).
  - `L2h13`: 10-13-13-10, 429 parameters; `L2h21`: 10-21-21-10, 861 (2x).
  - M_16,10 has 432 parameters.
- Seeds 0, 1, 2 for each size (12 models).
- Training: alg_zoo's recipe (AdamW, wd .001, lr .04/sqrt(first width), 2^14 warmup, decay to .01x, batch 1024,
  cross-entropy), n_train = 2^30 as in the zoo. One thread per run, at most 3 runs at once.
- Accuracy: on 1e7 fresh inputs (fixed seed 20261018), with binomial SE, against M_16,10's .9531175.
  A reference truth for scoring uses >= 2e8 further samples per model (seed disjoint from all estimators).

### Exact / deductive analysis (stage 2)
- Region counts. For 1-layer nets the activation regions are the chambers of a central arrangement of h
  hyperplanes in R^10: at most 2·sum_{i<10} C(h-1, i) (1,391,720 for h = 22). I count (i) the exact number by
  incremental LP enumeration where affordable, (ii) the distinct patterns seen in 1e7 samples, and (iii) the
  mass-coverage count N_delta (regions needed to cover 1-delta of the Gaussian mass, estimated from samples) for
  delta = 1e-2..1e-5, to compare with the RNN's prefix counts (34/792/21,032/632,394 at k = 2..5; N_1e-5
  34/634/10,233/152,956). For 2-layer nets: regions after layer 1 and after layer 2 (the "per layer" growth).
- Exact accuracy. Attempt an exact computation (region enumeration + Gaussian masses of the correct part of each
  region). Expected obstacle, stated in advance: masses of 10-D polyhedral cones have no closed form, so "exact"
  means deterministic quadrature to a stated tolerance, and its cost is likely far above sampling budgets. If it
  is out of reach, this is reported as a scoped negative with its cost estimate.
- Exact 1-D walks: along a line x + s·e_j (all other coordinates fixed) the network is piecewise linear in s; the
  correct set is a finite union of intervals, integrated against N(0,1) in closed form (as `m16/lineint.py` does
  for the RNN). Also exact great-circle walks in a 2-plane (homogeneity makes the radius free).

### Scoring (stage 3)
- Panel: per architecture, the 3 trained seeds plus 9 noisy copies (Gaussian noise of s.d. sigma·RMS on each
  weight matrix, sigma = .05 as in `m16/arc_eval.py`, noise seed fixed) = 12 models per architecture.
- Budgets C in {1k, 4k, 16k, 64k} FPE, 8 independent repeats per (model, budget).
- FPE = one forward pass of the network being estimated, counted in FLOPs (2 per multiply-add, plus 1 per ReLU,
  plus the argmax/label work not counted for either side). The walk estimators charge every FLOP they do
  (breakpoint solves, per-piece argmax solves, erf calls counted as 20 FLOPs) and convert to FPE of the same net.
- F = MSE · C_eff / mean p(1-p), C_eff = max(C, 1k FPE) (project definition, TASKS Directive 8 amendment 2), with a
  bootstrap CI over models. Plain sampling must give F ~ 1 (control).
- Estimators: E0 plain MC; E1 exact 1-D coordinate walks at random base points (conditional MC, sampling-based);
  E2 exact great-circle walks in random 2-planes (sampling-based); and, if found, a deductive estimator (no random
  inputs). The same E1 code with the same cost accounting is run on M_16,10 and its 9 noisy copies, so the
  architecture comparison uses one method on both.

### Hypotheses and how they could fail
- H1 [speculative]: a 432-parameter 1-layer MLP is clearly less accurate than M_16,10 (<.90), because 2nd-argmax
  needs comparisons among all pairs and the RNN reuses its weights 10 times. Fails if any 1-layer seed reaches .95.
- H2 [speculative]: 1-layer region counts are ~1e6 (arrangement bound), i.e. ~1e7 times fewer than the RNN's
  extrapolated ~1e13, so exact region enumeration finishes on a CPU; exact masses are the bottleneck.
- H3 [speculative]: walk estimators reach F < 1 on the MLPs (the network is cheap to walk) but not on M_16,10.
  Alternatives to keep in mind: (a) any F gain comes from the task's tie structure, not the architecture (then the
  RNN should gain similarly under the same walks), (b) the MLP is cheaper to sample too, so the gain may vanish
  in F even if walks are cheaper in FLOPs. The cheapest separating test is the E1 run on both families.

## Method
Key files: `train.py` (training), `models/*.json` (weights). Further files are added as stages run.

## Status
Running: pre-registered; training started.

## Results
None yet.

## Compute accounting
Training: 12 runs x 2^20 steps x ~1.4 ms = ~5 CPU-hours (not counted in F; it is model construction).

## Claims
None yet.

## Alternatives not ruled out
See H3 alternatives above.

## Failure modes
- Bias-free MLPs may be a weaker family than biased ones; a biased variant is a possible follow-up.
- A single training recipe (the zoo's) may undertrain the MLPs; loss curves are logged in the model files.
