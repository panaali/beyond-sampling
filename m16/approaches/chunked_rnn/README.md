# T392: chunked-input RNN, depth (k = 1) versus width (k = 10)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

Owner: Claude child session_01GcE8ZjdR1c4eu64CzHuyve, branch `claude/t392-chunked-rnn` (owner idea 06:0x).

## Question

ARC's M_16,10 reads one input per step for 10 steps (k = 1, deep in time). The all-at-once network of T345 (one hidden
layer of 128 comparator-like units) is the other extreme (k = 10, wide). An RNN that reads k numbers per step for
10 / k steps (k = 1, 2, 5, 10) interpolates between them. How does the difficulty of ARC's challenge (estimate the
accuracy from the weights, at least as precisely as sampling at equal compute) change between depth and width?
Concretely, per k: the smallest width reaching >= .95, its fragmentation (gate patterns), and the figure of merit F of
exact great-circle walks (random and fixed Halton sets) against plain sampling at equal forward-pass cost.

## Pre-registration (written and pushed before any training run, 2026-10-19)

### Model and task
- Task: alg_zoo `2nd_argmax`: x ~ N(0, I_10), label = `x.argsort(-1)[..., -2]` (position of the 2nd-largest input;
  alg_zoo/tasks.py). Accuracy = P(argmax logits == label).
- Model `k{K}h{H}` (`train.py`): torch `nn.RNN(input_size=k, hidden_size=H, nonlinearity="relu", bias=False)`, input
  reshaped to (S = 10/k steps, k) with consecutive positions per chunk (step t reads x[(t-1)k : tk]), h_0 = 0, logits =
  `Linear(H, 10, bias=False)` of the final hidden state. k = 1 is alg_zoo's DistRNN (M_16,10 = k1h16); k = 10 has one
  step, so W_hh is unused and the net is exactly T345's D1wH MLP.
- Recipe = alg_zoo `training.train`: AdamW (weight decay .001), lr .04/sqrt(H), 2^14 warm-up steps, exponential decay
  to .01x, batch 1024, cross-entropy, n_train = 2^30 fresh samples, `torch.manual_seed(seed)`, CPU, one thread per run.
- FPE (one forward pass of that net, in FLOPs; 2 per multiply-add, 1 per ReLU): 2 (10 H + (S - 1) H^2 + 10 H) + S H. The
  first step's W_hh h_0 is skipped (h_0 = 0); for k = 1 the T345 convention (which charges it) is also reported.

### Step 1: width ladder per k
- k in {1, 2, 5, 10}; H rungs 16, 24, 32, 48, 64, 128 (extended with 192, 256 if 128 fails).
- Seeds 0, 1, 2. A rung passes only if all 3 seeds reach >= .95; so seed 0 is trained first and seeds 1-2 only if seed 0
  passes (a failing seed 0 already fails the rung). The ladder for a k stops at the first passing rung = the
  **simplest passing config** for that k. Higher rungs may be started speculatively for CPU use; they do not change the
  selection. Scheduler: `ladder.py`.
- Accuracy of every trained model on 1e7 fresh inputs (numpy, seed 392001) with binomial SE; the decision uses this
  number.
- Cross-check for k = 10: T345's 2^30 D1w64 / D1w80 / D1w96 / D1w128 accuracies (.60 / .66 / .97, .90, .97 / .99 x 3).

### Step 2: on the simplest passing config per k
- Gate patterns: distinct full gate patterns (all S steps) in 1e7 inputs (seed 392002, 64-bit hashes), per-step
  cumulative counts, and the share seen once.
- Panel: the 3 seeds plus 3 noised copies per seed (relative sigma .001 x RMS per matrix, noise rng seed 392), 12
  models, as T345 Amendment B. Truth per model: 1e8-sample MC (seed 392003); its SE is reported.
- Walks: exact great-circle walks. x(theta) = u cos theta + v sin theta; by homogeneity the circle is the two lines
  +-(u + t v) with Cauchy weight; the correct set on each is a finite union of intervals, integrated exactly.
  - Generic implementation for all k (`cwalks.py`): `walks.walk` from T345 (`ffn_control/walks.py`, copied unchanged with
    its sha256 asserted) with a chunked-RNN model class (pre-activations and slopes at each piece). Costs charged as
    T345: `F` = implemented (one forward-with-slope = 2 FPE per piece + target/envelope/erf work), `F_incr` = incremental
    cost model (after a gate flip at step t: a rank-1 update into step t+1, then a value+slope recompute of the later
    steps and the readout; a model, not implemented), `F_floor` = 4 x 10 FLOPs per piece + aux + 2 FPE per walk.
  - For k = 10 also T345's `fast1.py` (incremental, implemented; copied unchanged, sha256 asserted).
  - Validation before scoring: (i) my k = 10 model class reproduces `walks.MLP` and `fast1.circle` on T345's D1w128_s0
    weights (same values, 20 circles, |diff| < 1e-8); (ii) my k = 1 class reproduces `walks.RNN` on a k = 1 model.
- (a) Random great circles (sampling-class): F = mean_m(Var_m c_m) / mean_m p_m(1 - p_m) with a 95% bootstrap CI over the
  12 models (budget-free for C >> c_m), from >= 1000 circles per model (fewer if a circle costs > 2 s; stated). Direct
  checks at C = 4k and 16k FPE: walks pooled per model in consecutive groups of floor(C / c_m) circles, each group one
  estimate, MSE against the 1e8 truth, F_direct = MSE C / p(1-p).
- (b) Fixed scrambled-Halton circle sets (deterministic, QMC-class; ARC acceptance unknown): `scipy.stats.qmc.Halton(d=20,
  scramble=True, seed=s)`, s = 1..5, first n points q, (u, v) = (Phi^-1(q_1..10), Phi^-1(q_11..20)) as T387; the same
  circles for every model; n = max(1, round(C / c_bar)) where c_bar is the mean implemented cost per circle on that
  k's panel from run (a) (no truths used); C = 4k and 16k FPE; cost charged as executed, C_eff = max(cost, 1k). F per
  set = mean_m e_m^2 C_eff / mean_m p(1-p); report the 5 sets and their mean; "Halton keeps F < 1" only if all 5 do.
  Done with the generic walk for every k, and with `fast1` for k = 10.
- Deliverable: a table and plot of F against k (generic F, F_incr, F_floor, Halton; fast1 at k = 10), pieces per circle,
  gate patterns, and a reading of depth versus width.

### Hypotheses (and how they could fail)
- H1 [speculative]: the width needed for .95 grows with k (k = 1 needs ~24-32, k = 10 needs 128, T345), because the
  recurrence reuses weights over steps. Fails if some k > 1 passes at a lower H than k = 1.
- H2 [speculative]: under the same generic walk code, F falls as k grows (fewer steps to propagate each flip), from
  ~15 (M_16,10, T345) toward ~9 (D1w128 generic, T345); F_incr falls more steeply. Alternative A: wider nets have more
  pieces per circle, so F is flat or U-shaped in k. Alternative B: F tracks accuracy (errors near ties integrate
  exactly; T345 saw F fall as accuracy rose), so differences in reached accuracy, not depth, drive F. Separating test:
  compare F at matched accuracy where two k's have rungs near the same accuracy, and report F per model against its
  accuracy within each k.
- H3 [speculative]: only k = 10 reaches F < 1 with an implemented walk (fast1's closed-form crossing angles exist only
  at depth 1); for k < 10 the generic walk stays above 1 and so does Halton.

## Method
Key files: `train.py` (training), `ladder.py` (scheduler + 1e7 accuracy), `cwalks.py` (chunked-RNN walk model class
+ validation), `score.py` (panel, truths, random and Halton F), `patterns.py` (gate patterns), `ref/` (T345 code
copied unchanged, hashes in `ref/SHA256SUMS`).

## Status
Pre-registered; training not started.

## Results
(none yet)

## Compute accounting
Training (model construction) is not counted in F. Walk costs are charged in FLOPs and converted to FPE of the net
being estimated (above). Truths: 1e8 forward passes per panel model.

## Claims
(none yet)

## Alternatives not ruled out
(to be filled)

## Failure modes
(to be filled)
