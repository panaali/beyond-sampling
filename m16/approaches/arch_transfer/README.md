# T432: which affordable, more modern architectures keep at least one of our positive methods?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · owner request: [OWNER_QA 6.10](../../../OWNER_QA.md#q6-10)
(related: [6.6](../../../OWNER_QA.md#q6-6) other tasks, [6.7](../../../OWNER_QA.md#q6-7) larger RNNs, T421 grid on branch claude/t421-scale-study)

**Pre-registration: committed 2026-10-19 before any training, LR screen or scoring.** Sections marked *(filled later)*
are written after the runs; nothing above them changes after training starts except by a dated, flagged amendment.

## Question

ARC's M_16,10 is a 16-unit bias-free ReLU RNN (432 weights) reading 10 N(0,1) numbers and outputting the position of
the 2nd largest (accuracy .953). Which affordable (<= ~20k parameters) more modern architectures, trained on the SAME
task, keep at least one of the project's positive methods, how far does each method get against both bars (strict
F <= 1, lenient F <= 10, [PROTOCOLS Directive 8](../../../docs/PROTOCOLS.md)), and which architectural property is
what decides it?

## Methods under test and the property each relies on (from the INDEX rows and code)

| id | method (INDEX rows) | what it computes | property it relies on |
|---|---|---|---|
| M1 | walk quadrature (H-E16, H-A3, T429) | P(wrong) = sum over tie faces of the exact misclassified 1-D mass along segments y + t(e_a - e_b)/sqrt2 (co-area identity, model-free), on a fixed Sobol set, sign strata, deterministic Neyman | (a) the restriction of the logits to a line can be split EXACTLY into pieces with known argmax (piecewise linear: roots of linear functions); (b) the radial integral (Student-t weight, hom.py) needs the argmax to be invariant under x -> r x, r > 0 (positive homogeneity: no biases, no mixed degrees); (c) the face law and windows are task-only |
| M2 | rare-event exact walks (H-R1) | far-gap event E1(t0) by exact gate walks along v3 lines + K-node exact-law transverse integration | as M1 (a); event and its law are task-only; accuracy depends on how thin the error slivers are |
| M3 | locate and repair (H-R2, H-R3) | weight-gradient of a rare-event probability (line-CMC boundary terms), accuracy curvature, a k = 12 weight edit, verified by conditional sampling against random edits | differentiability of the logit margin in the weights (a.e. for ReLU); the line-CMC gradient additionally needs exact crossings along lines (M1 (a)) |
| M4 | certified pruning, exact re-verification, rigorous polytope masses (T398, T400, H-C1) | branch and bound over step-wise gate patterns with LP (triangle) relaxations, re-verified LP duals, rigorous Gaussian mass of polytope cells | regions of constant activation pattern are POLYHEDRA (piecewise linear network), a sequential prefix (steps) to prune early, LP relaxations of each nonlinearity |
| M5 | safe-use rating (T401) | r(t) = P(3rd largest picked given g23 >= t) by M2's estimator with exact P(condition) | inherits M2 |

## 1. Applicability matrix (pre-registered predictions, [inferred]; tested cells are marked in Results)

Key: **E** = stays exact; **E-poly** = exact with polynomial roots (no longer linear); **A** = becomes approximate
(numerical 1-D quadrature along lines; CROWN / interval-style linear bounds for certificates: sound but loose);
**F** = fails as built. "rad" = the radial (homogeneous) integral of M1.

| feature | M1 walk quadrature | M2 rare-event walks | M3 locate and repair | M4 certified pruning + polytope mass | M5 rating |
|---|---|---|---|---|---|
| biases (ReLU) | E, but rad lost: integrate the Gaussian window directly (fewer effective lines per FPE) | E | E | E (polyhedra stay polyhedra); cone structure lost | as M2 |
| depth (stacked ReLU layers / RNN layers) | E (more kinks per line) | E | E | E; more patterns, triangle relaxations compound (looser) | as M2 |
| residual connections (additive) | E; rad kept if every branch is bias-free and degree 1 | E | E | E | as M2 |
| gated recurrence, HARD gates (hard-sigmoid / hard-tanh) | **A in general**: gate x state is a PRODUCT, so between kinks h_t is a polynomial in t whose degree can multiply every step (up to ~3^T); exact only where gates are saturated (0/1). **Correction to the DRI's 6.10 answer**, which listed hard gates as piecewise linear | A | E | F as built (products: non-convex regions); McCormick-relaxed bounds possible (sound, loose) | as M2 |
| gated recurrence, smooth gates (GRU / LSTM) | A | A | E | A (CROWN-style bounds only) | as M2 |
| linear diagonal state-space recurrence (non-selective, S4/LRU style) + ReLU MLP readout | E; rad kept if bias-free | E | E | E; no step-wise gate prefix in the recurrence itself (kinks only in the MLPs) | as M2 |
| selective SSM (Mamba: input-dependent decay / gates) | A (products, exp) | A | E | A | as M2 |
| 1-D convolution + ReLU + max-pooling | E (max-pool kinks are pairwise crossings); rad kept if bias-free | E | E | E (max is PL; relaxations exist); no sequential prefix: pruning is per layer | as M2 |
| attention with ReLU scores (relu(q.k)) | **E-poly**: scores are quadratic in t, outputs cubic per layer, degree ~3^L for L layers; rad lost (mixed degrees with the residual stream) | E-poly | E | F as built (semi-algebraic regions); McCormick-relaxed bounds possible | as M2 |
| softmax attention | A (exp / normalisation) | A | E | A | as M2 |
| normalisation (LayerNorm / RMSNorm) | A in general (division by a norm: rational); exception: in a bias-free chain without a residual bypass the norm is a positive per-input scale that commutes with ReLU and linear maps, so kinks stay where they were (E); BatchNorm at inference is affine (E) | as M1 | E | A / F (rational); the bias-free chain case is E | as M2 |
| smooth activations (GELU, SiLU, sigmoid, tanh) | A | A | E | A | as M2 |

Predicted class (the answer to 6.10, to be tested): **M1/M2/M4 stay exact exactly on the piecewise-linear class**
(ReLU / leaky ReLU / max / hard clipping, biases allowed, depth, residuals, convolution and max-pooling, linear
non-selective state-space recurrences, BatchNorm at inference); M1/M2 extend exactly to piecewise-polynomial nets
(ReLU attention) at higher root-finding cost; **M3 holds on every differentiable architecture**; **any multiplicative
interaction between two input-dependent quantities (gates, attention, selectivity) or a smooth nonlinearity removes
exactness**. The radial speed-up of M1 additionally needs positive homogeneity (bias-free, single degree).

Two alternative explanations to keep in mind for any per-architecture result (rule 3):
(i) a method may "fail" on a candidate because of accuracy/geometry (e.g. a 99%-accurate model has thinner error
slivers along lines), not because of the architectural feature: checked by the bias-free ReLU RNN control at the same
width and accuracy (T421's H = 32 models) and by reporting cells per line and error-set fragmentation per candidate;
(ii) an approximate (numerical) walk may score well because its error is dominated by the cross-line average, not by
the line integral: checked by measuring the line-integral error against a 1024-point reference on 2,000 lines.

## 2. Candidates (all on the same task: x ~ N(0, I_10), label = argsort(x)[-2], 10-way logits)

Fixed now; each has <= 20k parameters (exact counts in Results). No positional embeddings anywhere: the
transformer and the conv net are position-equivariant and give each position's logit from its own token, which the
task (permutation-equivariant) allows.

| tag | architecture | PL? | homogeneous (rad)? | features exercised |
|---|---|---|---|---|
| rnn32 (control, not trained here) | bias-free ReLU RNN H = 32 (T421 cpu seeds 0-2, alg_zoo DistRNN) | yes | yes | width only |
| rnnb32 | ReLU RNN H = 32 with biases (input, recurrent and output bias) | yes | no | biases |
| rnn2r32 | 2-layer stacked bias-free ReLU RNN, H = 32 each, logits = W_out (h1_T + h2_T) (residual readout) | yes | yes | depth, residual |
| hgru32 | GRU, H = 32, hard-sigmoid gates (clip(x/6 + 1/2, 0, 1)) and hard-tanh candidate, biases | no (gate x state products) | no | hard gates |
| sgru32 | the same GRU with sigmoid gates and tanh candidate (smooth twin) | no | no | smooth gates |
| ssm | bias-free 2-block diagonal linear SSM (N = 32, decay a = exp(-exp(nu)), LRU-style, non-selective), per-step ReLU MLP 32-64-32 with residual between blocks, ReLU MLP readout 32-64-10 from the last state | yes | yes | linear state-space recurrence, residual |
| conv | bias-free 1-D conv net: 4 conv layers (k = 3, 32 channels, dilations 1, 2, 4, 8, zero padding) + ReLU, global max-pool, per-position ReLU head [f_t; maxpool] 64-32-1 | yes | yes | convolution, max-pooling |
| tfrelu | 2-layer transformer, d = 32, 2 heads, token embedding ReLU MLP 1-32-32 (with bias), attention scores relu(q.k / sqrt(d_h)) / T, MLP 32-64-32 (bias), residual stream, no norm, per-token logit | no (polynomial, degree 9 along a line) | no | ReLU attention |
| tfsoft | identical except softmax attention | no | no | softmax attention |

Normalisation layers and smooth activations other than the GRU's are covered by the matrix only (no candidate),
unless time allows an optional `tfsoftln` (tfsoft with pre-RMSNorm), which would be added by amendment.

**Training recipe (zoo, as T86 / T421).** Own loop `train.py` that reproduces `alg_zoo.training.train` exactly
(AdamW, weight decay 1e-3, lr = lr_base / sqrt(width), 2^14 warm-up steps, exponential decay to 0.01x by mid-run,
batch 1024, fresh `th.randn(1024, 10)` per step, cross-entropy, `th.manual_seed(seed)`), with the model class swapped.
Check before use: for the bias-free DistRNN, `train.py` and `alg_zoo.training.train` give bit-identical weights after
2^18 samples (H = 16, seed 3). Width = H (RNNs), N (ssm), channels (conv), d (transformers).
- **LR screen (only tuning allowed):** lr_base in {0.04, 0.012, 0.004}, seed 100, n_train = 2^24 with the same
  schedule shape; keep the lr_base with the best 2^20-sample validation accuracy. Screen seeds are not panel models.
- **Length:** n_train = 2^30 if the measured single-thread step time projects <= 8 h per run, else 2^28 (same schedule
  shape), decided from timing before any full run and recorded.
- **Seeds:** 0, 1, 2 per candidate (the panel). All trained models are kept and reported, whatever their accuracy.
- **Accuracy:** plain MC, 1e7 N(0,1) inputs (`np.random.default_rng([432, 10])`), float64; truths for F use 1e8 inputs
  (`default_rng([432, 11])`), s.e. reported.

## 3. Scoring (fixed now)

**M1, walk quadrature.** The frozen T10 recipe (T429 `frozen/estimator.py` fb7902e12240: co-area tie-face identity,
sign strata, Sobol rule, 25% pilot then deterministic Neyman) with its walk replaced by my generic walk `walkgen.py`:
- PL candidates: exact piece-wise linear walk (value + tangent forward, next kink = smallest root of every ReLU /
  clip / max pre-activation, argmax within a piece from the logit roots). Exactness check: on M_16,10, `walkgen`
  line masses equal T429's `walk.py` to <= 1e-9 on 4,096 lines.
- tfrelu: exact piece-wise polynomial walk (truncation-free Taylor arithmetic of degree 9, roots by companion
  eigenvalues, each root polished and checked by evaluation); compared on 2,000 lines with the numerical walk at 1024
  points.
- hgru32, sgru32, tfsoft: numerical 1-D quadrature: correctness evaluated on a grid of G = 32 points uniform in the
  1-D CDF of the segment, every sign change refined by 12 bisection steps, mass between crossings exact from the CDF.
  Reported as approximate; its line-integral error vs a 1024-point reference on 2,000 lines is reported.
- Homogeneous candidates (rnn32, rnn2r32, ssm, conv) use the radial (Student-t) integrand; the others the Gaussian
  window (the recipe's `radial=False` path, tau capped at 8.5).
- **Cost:** every multiply-add and nonlinearity the walk executes is counted (instrumented arrays), divided by the
  FLOPs of one plain forward pass of THAT candidate (its own FPE, the unit plain sampling pays). No prefix saving
  (conservative vs T429).
- **Budgets:** 64k, 1e6, 4e6 FPE (T429's). **Rules:** RULE_SEED 31415 + 1000 r, r = 0..3 (4 fixed rules per model, to
  average the fixed-rule error over rule choice; each (model, rule) is one deterministic estimator run).
- **F** = mean over (model, rule) of [(est - p)^2 - se_truth^2] x C / mean_m p(1-p) (T429 definition, C = executed
  cost). CI: bootstrap over models (n = 3, weak; stated) with rules nested. **Bars:** strict "indicated" if point
  F <= 1 and F <= 1 for each of the 3 models' rule-averaged F; lenient likewise with 10. A "cleared" verdict needs a
  larger panel (proposed per-architecture follow-ups), not claimed here.
- Reference rows: M_16,10 (orig) and the rnn32 control are scored by the same code.

**M3, locate and repair (one model per candidate: seed 0).** Generic, autograd-based (works for any differentiable
net; the T372 line-CMC gradient is not ported):
- Event E1(t0): the model picks the rank-3 position while g23 = v2 - v3 > t0. t0 per model = the smallest value in
  {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5} with pilot P(E1(t0)) <= 1e-6 (1e7 conditional slab draws, T410's slab sampler
  re-derived; the conditional law given g23 > t0), and at least 50 pilot events; if none qualifies, the value with
  P closest to 1e-6 from above among those with >= 50 events is used and flagged.
- Locate: g_E = gradient in the weights of the smoothed event probability E_slab[w sigmoid((L_pos3 - max_{j != pos3}
  L_j) / tau)], tau = 0.05 x median |margin| on the design draws (1e6 design slab draws, seed 432100). S = the k = 12
  weights with the largest |g_E| (reported with the units they belong to).
- Repair: H_S = Hessian of the training loss (cross-entropy) restricted to S on 2^17 plain inputs (seed 432101);
  direction d = -(H_S + lam I)^-1 g_S, lam = 1e-3 trace(H_S)/12, unit L2. Step: eps in 0.001 x 2^j (j = 0..8), plus
  bisection; take the smallest eps whose design estimates (event count on 1e7 slab draws seed 432102, accuracy on 1e7
  plain inputs seed 432103, common random numbers) give P <= P0 / 20 and accuracy drop <= .02 pp; else the eps with
  the lowest design P meeting the accuracy limit (flagged).
- Verification (independent seeds, paired): ratio on 2e8 slab draws (seed 432200), accuracy change on 1e8 plain
  inputs (seed 432201). **Success (T402 / 6.7 criterion):** ratio <= .1 with its 95% upper end <= .15 AND accuracy
  drop <= .05 pp. **Control:** 20 random edits of the same 12 weights at the same L2 norm (numpy default_rng(432300)),
  each verified on 2e7 slab draws and 2e7 inputs; **targeting matters** if <= 2/20 succeed and the targeted ratio is
  below the smallest random ratio.
- Cost reported in FPE (design + verification); plain-MC expected event count at the design cost reported.

**M4, certified pruning feasibility (PL candidates only).** Not a full certificate. Reported: (a) PL yes/no and
whether a step-wise prefix exists; (b) mean cells per line in M1's walks (fragmentation) vs M_16,10 and rnn32;
(c) the matrix verdict. **M2 / M5** are scored by the matrix only here; per-architecture sessions are proposed.

**Predictions (made now).** Accuracy: rnnb32, rnn2r32, ssm, conv, tfsoft >= .953; hgru32 / sgru32 >= .953; tfrelu
uncertain (may train worse than tfsoft). M1: PL candidates F within 2x of the rnn32 control at each budget, lenient
indicated for all, strict indicated at 1e6 and 4e6 for the RNN-family PL candidates; non-homogeneous ones worse than
homogeneous ones (no radial integral); numerical-walk candidates (hgru32, sgru32, tfsoft): lenient indicated, strict
not (the quadrature error per line adds bias). M3: success on >= 5 of 8 candidates, targeting matters on those.

## Method (code)
*(filled later)*

### Record R1 (2026-10-19, before any screen or panel run): length and threads
- Loop check passed: `train.py` and `alg_zoo.training.train` give bit-identical DistRNN weights after 2^18 samples
  (H = 16, seed 3; max |dW| = 0).
- Single-thread step times (ms, batch 1024): rnnb32 7.3, rnn2r32 13.2, hgru32 57.3, sgru32 52.1, ssm 20.2, conv 26.4,
  tfrelu 57.3, tfsoft 67.2. By the pre-registered rule (<= 8 h at 2^30): **2^30** for rnnb32, rnn2r32, ssm, conv;
  **2^28** for hgru32, sgru32, tfrelu, tfsoft (same schedule shape, 4x fewer steps).
- Threads per run (2 runs per container, rule 8): 2 for conv, tfrelu, tfsoft (measured 1.4-1.8x faster per step),
  1 otherwise. Thread count can change float reduction order; it is recorded in each model's json.
- Parameter counts: rnnb32 1,450; rnn2r32 3,424; hgru32 / sgru32 3,690; ssm 7,904; conv 11,392; tfrelu / tfsoft 17,729.
- Scripts: `archs.py` (models), `train.py` (recipe), `job.sh` / `arch.sh` (screen, pick, panel; commit per model),
  `launch.sh` / `wait.sh` / `setup.sh` (helper containers).

### Amendment A1 (2026-10-19 ~17:45, after the screens started, before any panel model finished)
- **Screen defect.** alg_zoo's warm-up is a fixed 2^14 steps; a 2^24-sample screen is exactly 2^14 steps, so it ends
  at the top of the warm-up and measures early progress, which favours the largest lr_base (rnnb32: .526 / .315 / .177
  for .04 / .012 / .004). The screens are kept as run (helpers were already using them); in effect they select the
  zoo default 0.04 unless a lower rate trains faster from the start. Recorded as a protocol defect.
- **Fallback (fixed now, before any panel accuracy exists):** a candidate whose seed-0 panel run diverges (NaN loss) or
  ends below 0.80 accuracy on 2^20 validation inputs is retrained (all 3 seeds) at the next lower screen rate; both
  sets are reported. Warm-up stays 2^14 steps in all panel runs (the zoo's absolute value, also at 2^28).
- **Repair sampler bug fixed before any repair run:** the slab weight lacked the proposal normaliser; after the fix,
  on M_16,10: P(g23 > 1) .04443 (T410 .04445), P(E1(1.0)) 5.56e-6 (T410 5.43e-6 +- .16e-6), P(E1(2.0)) 7.5e-9
  (4e6 draws, ~50 events; T364 6.81e-9).
- **Added reference rows:** M_16,10 (`m16`) is also run through M1 and M3 (repair) by the same code, as a check that
  the generic pipeline reproduces the known results (T429 / T402).

### Record A2 (2026-10-19 23:3x): fallback triggered for ssm; one bug fixed
- ssm (lr_base 0.04 from the screen) ended at 2^20 validation accuracy .765 / .759 / .756 (< .80), so by A1 it is
  retrained, all 3 seeds, at lr_base 0.012 under the tag `ssmB` (same architecture). Both sets are evaluated and reported.
- Bug fixed (walkgen.Net): the radial (homogeneous) integrand was switched on by network kind, so rnnb32 (an `rnn` with
  biases) was scored with it; those M1 rows were deleted unanalysed beyond a first table that showed the error
  (estimates ~.9903 vs truth ~.9968) and are rerun with the Gaussian window, as pre-registered. No other candidate
  was affected (all other biased kinds were already non-homogeneous).
