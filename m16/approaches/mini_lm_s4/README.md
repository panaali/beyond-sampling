# T447: Mini-LM stream S4: certified upper bounds on illegal outputs

Task row: T447 in [TASKS.md](../../../TASKS.md). Spec: [STREAMS.md S4](../mini_lm/STREAMS.md) (branch `claude/t433-mini-lm`
d45e9d21). Session: Claude child session_01CQSTQDBPTD6HNGaZP94s3c. Branch `claude/t447-mini-lm-s4`. Harness (read-only):
`../mini_lm/{load,models,langs,metrics}.py`.

## Pre-registration (written 2026-10-19, before any L = 32 run; dev work on the L = 15 baselines only)

**Target.** B_tau = P(max_t p_illegal(x, t) > tau) on dyck2 (tau in {0.5, 0.9, 0.99}); secondary modsum at the small
horizon only (soundness).

**What counts as a certificate.** A number UB with UB >= B_tau for the model evaluated in real arithmetic on its
float32 weights, with a margin DELTA = 1e-3 in log-odds on every decision (covers the harness's float32 evaluation
error, assumption A1 below). Deductive: no random model inputs.

**Methods** (all standard; we claim no novelty for any of them):
- **M1, prefix branch and bound with IBP closure** (`bab.py`, `cert.py`). Node = prefix y_<k with its exact law
  probability. A node is charged BAD (full mass) if the illegal log-odds bound at a concrete position reaches
  logit(tau) - DELTA; CLOSED if, for every later position, the bound over all continuations keeps every illegal set
  that position can have below that line; otherwise expanded. UB = mass(BAD) + mass(OPEN) = 1 - mass(CLOSED).
  Bounds: interval bound propagation (Gowal et al. 2018) over the convex hull of the allowed embeddings, made tighter
  by keeping each position's own token exact (a union of per-token boxes, hulled only where other positions read it).
  Attention: hard = hull of every candidate that can reach the row maximum (this is the "branch on routing" union,
  boxed); relu / linear = interval products of score and value boxes (McCormick-equivalent for one product); pos =
  exact fixed mixing; softmax = weight box intersected with the value hull. Illegal sets at abstract positions use
  the depth parity and the own token (an opener of type c fixes the illegal set to the other closer).
- **M2, the same with an affine-form (zonotope) domain** (`zono.py`): one generator per (abstract position, token)
  on a product of simplices (exact for linear maps), DeepZ ReLU relaxation (Singh et al. 2018), affine products as
  in DeepT (Bonaert et al. 2021). Built for the RNN; transformer only if M2 helps the RNN.
- **M3, stack-merged abstract interpretation (RNN only)** (`reach_rnn.py`): all prefixes with the same bracket stack
  (optionally the same last m tokens) are merged into one hidden-state box; only legal continuations are propagated;
  each step checks the exact illegal set of that stack. A one-pass bound for any horizon if the boxes stay tight.

**Dev / test split.** Recipe choices (start depth K0 of the branch and bound, M1 vs M2, m in M3, chunking) are tuned on
the 36 L = 15 baselines only, then frozen in this README before any MLP1 (L = 32) run.

**Soundness check (L = 15).** For every one of the 36 baselines and each tau: UB >= the exact truth
(`small_exact` in the baseline json; violation = UB < truth (1 - 1e-9)), AND a leaf-level check: no CLOSED prefix has a
reference-bad leaf below it (reference = the harness's `metrics.per_sequence` max_t p_illegal over every L = 15 sequence,
`ref.py`); for modsum (no closure possible without the running sum) every reference-bad sequence must be charged.
**Pass line: 0 violations of either kind.**

**Usefulness (L = 32, MLP1: 20 retrains per dyck2 architecture, seeds 100-119).** Groups = architecture (rnn, hard,
relu, linear, pos; softmax as reference) x tau. Cost charged in FPE = FLOPs actually performed by the bound code
(every matmul, centre and radius, multiply-add = 2) / `model.flops(32)`. Sampling bar: plain MC with N = budget
sequences (seed 1000 + model seed), the upper end of the two-sided 95% Clopper-Pearson interval (0 events at 1e6:
3.69e-6). **Strict:** UB <= CP95 at the same FPE; **lenient:** UB <= 10 x CP95; a group passes a bar if >= 50% of its
models do, at budget **1e6 FPE** (1e5 also reported). Primary: tau = 0.9 groups; all 18 groups reported.

**Predictions (dev evidence so far: M1 closes only 1-2 tokens ahead at L = 15; IBP radii grow ~5.7x per RNN step,
rho(|W|) = 5.7, and ~4x per step in the zonotope domain).**
- S1: 0 soundness violations at L = 15 (pass line).
- S2: at L = 15, M1 costs MORE than exact enumeration (7.4e6 FPE), within 10% of the truth when it finishes.
- S3: at L = 32 and 1e6 FPE, UB = 1 (nothing certified) for >= 90% of MLP1 models in every group: both bars FAIL.
  Reason: with closure m* tokens ahead the bound needs about count(31 - m*) ~ 1e12-1e13 nodes.
- S4: M3 does not certify the RNN (UB ~ 1 at L = 15).

**Exhaustion checklist** (variants tried / not tried) will be kept under Results.

## Recipe freeze for L = 32 (written 2026-10-19, after dev on the L = 15 baselines, before any MLP1 run)

Dev evidence (all on the 36 baselines; files in `data/`, reproduced by the commands under Results):
- **Certification horizon** (`horizon.py`): the fraction of sampled prefix nodes whose closure test passes for the next
  A positions. M1 (IBP, own token exact) passes A = 1 for 97-100% of nodes on every architecture and A = 2 for 0%
  (hard attention: 24% at L = 32), A = 3 for 0%; M2 (zonotope, RNN) passes A = 2 for 26-70%, A = 3 for 0%.
  A = 1 is no saving at all: with every earlier position concrete, the "abstract" next token is enumerated exactly,
  so M1 at L = 15 is exhaustive enumeration in disguise.
- **Why** (measured): RNN rho(|W|) = 5.7 (rho(W) = 2.1), so interval radii grow ~5.7x per step; the zonotope domain
  still grows ~4x per step (DeepZ with ReLU errors as generators: ~6x, worse); along real trajectories the RNN is only
  weakly contracting (top Lyapunov exponent -0.08 to -0.11 per step). The first abstract step already has logit
  radius 2.5-3.7: the union over continuations legitimately mixes states where ')' is the legal closer with states
  where ']' is, and a box (or zonotope) cannot keep "which closer is legal" tied to "which logit is high".
- **M3** (stack-merged boxes, RNN) gives UB 0.992-0.9998 at L = 15 for every m in {0, 2, 4} (truth 0): dead on arrival
  for the same reason (hidden-state spread for a fixed stack ~0.6 per unit, amplified by |W|).

**Frozen recipe (all architectures):** M1, depth-first branch and bound from the root (`bab.run_dfs`, largest mass first
within the deepest pool, chunk 128), closure tests only at nodes with A <= AMAX = 1 positions left (elsewhere only the
concrete check), DELTA = 1e-3, HARD_ETA = 1e-4. Budgets: tau = 0.9 to 1e6 FPE (snapshot at 1e5), tau = 0.5 and 0.99 to
1e5 FPE (`score32.py`). M2 is not used for scoring: it gains at most one position for a fraction of nodes (a <= 3x node
saving), while about 20 positions would be needed (below), and it costs more per test.

**Prediction sharpened (S3).** A node at depth j has probability <= 2^-j, and closure needs j >= 31 - A_max = 30, so a run
that performs n_t closure tests certifies mass <= n_t 2^-30. At 1e6 FPE (n_t <= ~1e5-4e5 tests) UB >= 0.9996 for every
model and every node order: both bars fail whenever CP95 < 0.1 (lenient) or < 0.9996
(strict), i.e. for every model with fewer than ~1e5 events in 1e6 samples. To match CP95 (~4e-6 at 1e6 FPE), the bound would have to close nodes at depth <= ~11 (count(11) ~ 1e5
nodes), i.e. certify ~20 positions ahead.

## Assumption A1

The harness evaluates the model in float32. Our bounds hold for real arithmetic on the float32 weights (float64
evaluation with a 1e-12 relative radius inflation per operation, >> float64 rounding of <= 64-term dot products), and
every decision keeps a log-odds margin DELTA = 1e-3 and, for hard attention, a score tie slack HARD_ETA = 1e-4 on top
of the harness's 1e-6. That these cover the float32 evaluation error is checked empirically (leaf-level check), not
proved.
