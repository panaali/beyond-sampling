# T433: mini language model harness (lead)

Owner request: [OWNER_QA 6.11](../../../OWNER_QA.md#q6-11) (and 6.10). Task row: T433 in [TASKS.md](../../../TASKS.md).
Session: Claude child session_01Jc3vr4TPRVxxwK9wJAkC1Y. Branch `claude/t433-mini-lm`.

## Pre-registration (written 2026-10-19, before any model was trained)

### (a) Languages

Both languages have an exactly known next-token law, a natural "bad output", a small horizon where every
sequence can be enumerated (exact ground truth), and a large horizon where enumeration is infeasible and
sampling is the real baseline.

**L1 `dyck2` (context-free; needs a stack).** Tokens `( [ ) ]` (ids 0-3) plus BOS (id 4, input only).
Generator, prefix-closed (no forced closing at the end, so the first t tokens of a length-L sequence have
exactly the length-t law): at depth 0 open `(` or `[` with prob 1/2 each; at depth d > 0 open each type with
prob q/2 and close the top-of-stack type with prob 1-q; q = 0.5; no depth cap. The model reads BOS x_1..x_{L-1}
and predicts x_1..x_L. Legal next tokens: both openers always; the matching closer iff d > 0.
*Bad output*: an illegal token, i.e. the WRONG closer (d > 0) or any closer on an empty stack (d = 0).
Horizons: large L = 32 (931e12 sequences with positive probability); small T_s = the largest t with at most
2e7 sequences, which is t = 15 (7,365,120 sequences; counts by DP in `langs.py`). Because the model is causal
and the law prefix-closed, the small horizon is the SAME model scored on its first 15 predictions.

**L2 `modsum` (regular; DFA with 5 states).** Digits 0..4 (ids 0-4) and `=` (id 5), BOS (id 6, input only).
Sentence `d_1 .. d_n = s` with n ~ Uniform{1..20}, d_i iid uniform, s = (sum d_i) mod 5; padded after s with
`=` (unscored). Exact next-token law: after k < 20 digits, `=` with prob 1/(21-k) (hazard of the uniform n),
else a uniform digit; after 20 digits `=`; after `=`, s with prob 1. *Bad output*: a wrong answer s' != s.
Horizons: large n = 20 (5^20 = 9.5e13 digit strings); small n <= 10 (sum 5^n = 1.22e7 strings, enumerated).

### (b) Harness

- `langs.py`: samplers (numpy, seeded), exact enumerators for the small horizons with exact probabilities,
  legal-token masks, exact next-token distributions (for NLL).
- `models.py`: ReLU RNN and tiny transformer with attention variants; parameter and FLOP counts.
- `metrics.py`: one function scores a model on a weighted batch (exact enumeration = weights are the exact
  probabilities; sampling = weights 1/N). Metrics, with p_illegal(x, t) = model probability mass on illegal
  next tokens at position t:
  - `acc`: dyck2: at positions with d > 0, P(model ranks the correct closer above the wrong closer)
    (the hard, memory-dependent part); modsum: P(argmax at the answer position = s).
  - `illegal_mass`: mean over scored positions of p_illegal (dyck2: all positions; modsum: answer position).
  - `nll_excess`: mean KL(true next-token law || model) over scored positions (nats).
  - rare bad output, per sequence: **B_tau(x) = 1[max_t p_illegal(x, t) > tau]**, reported for
    tau in {0.5, 0.9, 0.99}. tau = 0.5 means "greedy decoding emits an illegal token somewhere"; larger tau gives
    rarer, more confident failures (the analogue of E1(t0) on M_16,10).
  Ground truth: small horizon exact (float64 accumulation); large horizon 1e6-sample estimate for the
  baseline table (high-precision truths for estimator panels are each stream's job, at 1e8+ samples).
- Cost unit: 1 FPE = one forward pass of the model over one full sequence of the stated horizon
  (FLOPs per FPE reported per model, multiply-add = 2).
- `train.py`: AdamW lr 3e-3 (cosine to 0), weight decay 0.01 (not on embeddings), batch 256, fresh samples every
  step, 8000 steps (dyck2) / 8000 steps (modsum), grad-clip 1.0, torch threads 2, seeds 0, 1, 2. Loss: cross-
  entropy on every next token (dyck2) / on all tokens up to and including s (modsum), against sampled targets.
  Checkpoints `models/<lang>_<arch>_s<seed>.pt` (state dict + config), metrics json next to each.
- `load.py`: `load(name) -> (model, lang)`.

### (c) Baselines (all <= ~20k parameters, 3 seeds each, both languages)

- `rnn`: h_t = relu(W h_{t-1} + U[x_t] + b), H = 64, logits = V h_t + c.
- tiny transformer, d = 32, 2 layers, 1 head, ReLU MLP (width 64), residual, learned token and position
  embeddings, NO layer norm, causal mask. Attention variants (the only difference):
  - `softmax` (reference);
  - `hard`: average-hard attention (uniform over the argmax set of q.k, as in Hao-Angluin-Frank 2022 AHAT),
    trained straight-through (forward hard, backward softmax at the same scores); the network is piecewise
    linear given the routing, routing piecewise constant;
  - `relu`: weights relu(q.k / sqrt(d)) / t (Wortsman et al. 2023 style; piecewise polynomial);
  - `linear`: weights (q.k / sqrt(d)) / t (polynomial, no normalisation by a softmax);
  - `pos`: position-only, weights softmax(A[t, :t]) with a learned A (input-independent linear mixing; the whole
    network is piecewise linear in the embeddings).
  6 architectures x 2 languages x 3 seeds = 36 models.

### Pre-registered predictions (scored when the baselines finish; wrong predictions stay in the record)

- P1: softmax reaches dyck2 `acc` >= .95 at L = 32 and modsum `acc` >= .90 at n = 20 (median over seeds).
- P2: `hard` is within 2 pp of softmax on both languages (median).
- P3: `relu` and `linear` are within 5 pp of softmax.
- P4: `pos` is at least 10 pp below softmax on dyck2 (it cannot route to the matching bracket by content).
- P5: the rnn is the best model on modsum (a finite-state rule).
- P6: for at least one exactness-friendly variant (hard, relu, linear, pos, rnn), B_0.9 at the large horizon lies
  in [1e-6, 1e-2] (rare but measurable: the regime where rare-event methods matter).
- P7 (sanity): exact small-horizon metrics match a 1e6-sample estimate within 4 s.e. for every model.

Literature used for the variants (all pre-2026, so no web-log entry needed): Hahn 2020 (hard attention limits),
Yao et al. 2021 (self-attention on Dyck), Hao, Angluin and Frank 2022 (AHAT), Katharopoulos et al. 2020 (linear
attention), Wortsman et al. 2023 (ReLU attention), Shi et al. 2020 and Bonaert et al. 2021 (transformer
verification), Xu et al. 2020 (auto_LiRPA / CROWN).

---

Proposed parallel streams: [STREAMS.md](STREAMS.md).

## Question

Do the project's positive methods (exact walks / walk quadrature, rare-event walks, locate and repair, safe-use rating,
certified bounds) carry over to a tiny next-token LANGUAGE model, where inputs are discrete tokens and attention may be
non-piecewise-linear? This task (the lead) builds the shared languages, harness and baselines and proposes the
parallel streams; the streams answer the question.

## Method

Pre-registered above (commit 43321125, before any training). Files:

| File | Purpose |
|---|---|
| `langs.py` | `Dyck2`, `ModSum`: samplers, exact enumerators (`enumerate_small`, streaming `enumerate_chunks`), `annotate` (legal masks, exact next-token law, depth / answer) |
| `models.py` | `ReLURNN`, `TinyTransformer(attn=softmax/hard/relu/linear/pos)`, `flops(T)` (multiply-add = 2) |
| `metrics.py` | `per_sequence`, `score_chunks` / `score` (weighted: exact or sampled), `exact_small`, `sampled` |
| `train.py` | training + scoring of one baseline (`--eval_only` re-scores a checkpoint) |
| `run_baselines.sh` | all 36 baselines, 2 queues, `ulimit -v 2500000` |
| `load.py` | `load("dyck2_hard_s0") -> (model, lang)`; `names()` |
| `summarize.py` | -> `results_baselines.md` |
| `models/` | 36 checkpoints (`.pt`: cfg + state dict) and their metrics (`.json`) |

Usage: `source .venv/bin/activate` (CPU torch 2.14; the venv was created in this container with
`pip install torch --index-url https://download.pytorch.org/whl/cpu numpy scipy`), then from this folder
`python -c "from load import load; m, lang = load('dyck2_relu_s1')"`.

## Status

**Done (harness and baselines).** Deviations from the pre-registration, all disclosed:
1. Exact `dyck2` scoring first ran out of memory under the 2.5 GB cap (int64 copy of 7.4e6 x 15); fixed by streaming
   int8 chunks (commit 07e37d23). The dyck2 checkpoints were trained by the first run and scored by `--eval_only`, so
   their json has `train_seconds: null` and no loss log (the weights are the pre-registered recipe, unchanged).
2. `dyck2_hard_s1` scoring hit the memory cap in the transformer forward; rerun with forward chunk 8192 (no other change).
3. The first 2-h runner stopped at its time limit; the rest ran in a second invocation (skips finished models).

## Results

Full table: [`results_baselines.md`](results_baselines.md) (`python summarize.py`; median [min, max] over seeds 0-2;
"large" = 1e6 fresh samples at L = 32 / n = 20; "small_exact" = exact enumeration of L = 15 / n <= 10).
Reproduce everything: `./run_baselines.sh` (about 4 h on 4 cores). Condensed (medians):

| lang | arch | params | acc large | B0.5 large | acc small exact | B0.5 small exact | B0.9 small exact |
|---|---|---|---|---|---|---|---|
| dyck2 | rnn | 4,740 | 1.000 | 1.4e-4 | 1.000 | 0 | 0 |
| dyck2 | softmax | 17,956 | 1.000 | 3e-6 | 1.000 | 0 (max 4.5e-7) | 0 |
| dyck2 | hard | 17,956 | .928 | 1.6e-3 | .966 | 2.9e-4 | 0 |
| dyck2 | relu | 17,956 | 1.000 | 0 (max 2.9e-3) | 1.000 | 0 (max 2.6e-5) | 0 |
| dyck2 | linear | 17,956 | 1.000 | 2.0e-4 | 1.000 | 1.5e-6 | 0 (max 6e-8) |
| dyck2 | pos | 15,908 | .996 | 6.4e-4 | 1.000 | 1.1e-5 | 0 |
| modsum | rnn | 4,998 | 1.000 | 0 | 1.000 | 0 | 0 |
| modsum | softmax | 17,766 | .200 (chance) | 1 | .959 | .058 | 1.9e-3 |
| modsum | hard | 17,766 | .199 | 1 | .38 | .78 | 1.2e-3 |
| modsum | relu | 17,766 | .200 | 1 | .28 | .90 | 0 |
| modsum | linear | 17,766 | .987 (s0 chance) | .019 | 1.000 | 1.4e-4 | 2.0e-5 |
| modsum | pos | 14,638 | .593 (s1 .983) | .60 | .999 | 1.0e-3 | 1.5e-4 |

FLOPs per FPE (large horizon): dyck2 transformer 1.19e6 (pos 8.6e5), rnn 2.8e5; modsum transformer 7.9e5, rnn 2.0e5.
Exact small-horizon scoring costs about 4-5 CPU-min per model (7.4e6 / 1.2e7 sequences).

Prediction scores:
- P1 **partly wrong**: dyck2 softmax acc 1.000 (yes); modsum softmax .200 = chance at n = 20 (no). The modsum
  transformers learn short sums (softmax small-horizon acc .959) but not n = 20 under the 8000-step recipe; only
  linear s1/s2 and pos s1 reach >= .98. The RNN learns modsum perfectly.
- P2 **wrong**: hard attention is 7 pp below softmax on dyck2 (.928 vs 1.000; straight-through training).
- P3 holds on dyck2 (relu 1.000, linear 1.000; relu s1 is weaker: acc .993, B0.5 2.8e-3); on modsum it holds only
  literally (softmax itself is at chance).
- P4 **wrong**: position-only attention reaches .996 on dyck2 (only .4 pp below softmax), against the predicted >= 10 pp
  gap. [inferred] With 2 layers + MLP, a fixed causal mixing of all previous tokens carries enough of the bracket
  history at L = 32; its errors are 200x more frequent than softmax's (B0.5 6.4e-4 vs 3e-6).
- P5 holds (rnn acc 1.000 on modsum, all seeds).
- P6 holds: B0.9 at the large horizon is in [1e-6, 1e-2] for modsum linear (2.2e-3) and pos (1.1e-3, 2.9e-5); on dyck2
  B0.9 is < 1e-6 (0 of 1e6) for most models, i.e. the dyck2 rare events at tau >= .9 are rarer than plain sampling at
  1e6 can see: the target regime for rare-event methods.
- P7 holds: exact small-horizon B0.5 / B0.9 vs a 1e6-sample estimate, max |z| 2.31 over 36 models (< 4).

## Compute accounting

Training: 8000 steps x 256 sequences = 2.05e6 sampled sequences per model, 85-135 s on 2 threads. Scoring per model:
1e6 large-horizon sequences + 1e6 small-horizon sequences + the full small-horizon enumeration (7.37e6 dyck2 /
1.22e7 modsum sequences). Total for 36 models about 6 CPU-h (2 processes x 2 threads, about 4 h wall including the
memory-failure reruns). No GPU.

## Claims

- [measured] All 36 baselines (<= 18k params) train on CPU in about 2 min; dyck2 is learned to >= .996 closer
  accuracy at L = 32 by every architecture except hard attention (.928).
- [measured] Exact enumeration at the small horizon agrees with sampling (P7) and costs 4-5 CPU-min per model.
- [measured] Under the pre-registered recipe the modsum transformers fail at n = 20 (11 of 15 at chance, pos s2 .59, only linear s1/s2 and pos s1 >= .98); the RNN does not.
- [inferred] dyck2's illegal-output events at tau >= .5 range from about 1e-6 (softmax) to 1e-3 (hard, pos): rare enough
  that plain sampling at 1e4-1e6 FPE sees few or none, so the rare-event streams have a real target.
- [speculative] Position-only and linear attention models (piecewise linear / polynomial in the token embeddings)
  are accurate enough on dyck2 to be the main vehicles for exact methods.

## Alternatives not ruled out

- Hard attention's 7 pp gap may be the straight-through training, not the architecture (annealed temperature,
  sparsemax or a tie tolerance may close it): S1.
- The modsum failure may be the step count or curriculum (parity-like targets are known to plateau under SGD), not
  capacity: S1. Seeds that escaped (linear, pos) show it is reachable.
- Accuracy saturating at 1.000 makes acc a poor estimator target on dyck2; illegal mass and B_tau are the
  informative quantities.
- 3 seeds only; the stream panels should be >= 20 retrains per architecture.

## Failure modes

- `hard` attention ties: the tie tolerance 1e-6 rarely triggers with learned position embeddings, so it behaves as
  unique-hard attention (no uniform averaging).
- Memory: the transformer forward on 32k x 32 hard-attention masks exceeded 2.5 GB; keep forward chunks <= 8192.
- Exact enumeration grows about x3 per position (dyck2: 2.2e7 at L = 16, 1.8e9 at 20, 9.3e14 at 32).
