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
