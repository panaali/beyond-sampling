# T433: proposed parallel streams (lead's specs for the DRI)

Harness, baselines and numbers: [README.md](README.md) (branch `claude/t433-mini-lm`). Every stream works in its own
folder `m16/approaches/mini_lm_<stream>/`, imports the harness read-only (`sys.path` to `m16/approaches/mini_lm`),
and pre-registers before touching truths.

## Shared conventions (all streams)

- **Models.** Development: the 36 baselines (`load.names()`). Scoring: panel **MLP1** = 20 fresh retrains per dyck2
  architecture {rnn, softmax, hard, relu, linear, pos}, seeds 100-119, built with `train.py` unchanged (S2 builds it
  first and pushes it; about 4 CPU-h training, 120 x 2 min). modsum is secondary (only rnn, linear s1/s2, pos s1
  are trained to n = 20; S1 may supply a working recipe).
- **Cost.** 1 FPE = one forward pass over one full sequence at the stated horizon (`model.flops(L)` FLOPs, e.g.
  1.19e6 for a dyck2 transformer at L = 32). Charge every weight-dependent operation at those rates
  (multiply-add = 2); prefix-tree methods are charged per node actually computed (an RNN step = 1/L FPE; a
  transformer step at depth t with a KV cache = its own FLOPs). Model-free tables over the language law
  (e.g. prefix counts, exact token probabilities) are free if precomputed once and listed. Floor C_eff = max(C, 1k FPE).
- **Targets.** Q_ill = E[mean_t p_illegal] (illegal mass), Q_acc (closer accuracy, saturated near 1 on dyck2 so a
  secondary target), and the rare events B_tau = P(max_t p_illegal > tau).
- **Bars (Directive 8).** For a target with plain-MC per-FPE variance sigma_m^2 (measured once per model on 1e6
  samples) and errors e_m on the panel: F = mean_m(e_m^2) * C_eff / mean_m(sigma_m^2); for rare events
  F_rel = mean_m[e_m^2 * C_eff / (p_m (1 - p_m))]. **Strict F <= 1** (point; CI reported, "significant" if the
  upper 95% bootstrap CI over models <= 1) and **lenient F <= 10** (upper CI), each per architecture group, at ONE
  stated budget in {1e3, 1e4, 1e5, 1e6} FPE, every step labelled deductive or inductive.
- **Truths.** Small horizon (L = 15): exact enumeration (`metrics.exact_small`, 4-5 CPU-min per model). Large
  horizon (L = 32): plain MC at >= 100x the largest budget scored, or an exact method validated at L = 15.
- **Compute rules.** CPU first; 2 heavy processes per container, `ulimit -v 2500000`; transformer forward chunks
  <= 8192 sequences.

## S1: exactness-preserving attention, trained to softmax accuracy

**Spec.** Build attention variants that keep at least one exact tool and match softmax on dyck2 (and repair the
modsum recipe). Candidates: hard attention trained by temperature annealing instead of straight-through, or with a
tie tolerance so it can average (AHAT); sparsemax / entmax-1.5 (piecewise linear in the scores); ReLU attention
with fixed 1/t normalisation (already .99-1.0); position-only attention (already .996: try 1 layer, gated
position mixing, and position mixing plus one hard head); key-only scores (q constant), so routing depends only on
the key tokens. For modsum: curriculum on n, more steps, lr sweep, until at least one exactness-friendly variant
reaches >= .95 at n = 20. Deliver frozen recipes plus 3 seeds each, and a table of which exact tool each variant
keeps (piecewise linear / piecewise polynomial / smooth).
**Inputs.** `models.py` (subclass `Block`, do not edit), `train.py` recipe, baselines for reference.
**Success (pre-registered).** dyck2: a variant whose routing is piecewise constant or linear (hard, sparsemax,
pos, key-only) is within 1 pp of softmax's closer accuracy at L = 32 and within 10x of softmax's B0.5, on 3 of 3
seeds; modsum: acc >= .95 at n = 20 for one exactness-friendly variant on 2 of 3 seeds. This stream builds models;
it scores no estimator, so the F bars apply downstream.
**Compute.** About 40 trainings x 2-4 min plus scoring, about 6-8 CPU-h; one container.

## S2: estimators for discrete inputs vs sampling at equal compute

**Spec.** Estimate Q_ill (and Q_acc) at L = 32 for every MLP1 model. Methods, in order of expected value:
(a) *exact head, sampled tail*: enumerate every prefix up to depth t0 (t0 = 10-15: 3e4-7e6 nodes, computed once as
a tree, so each node costs one step) and sample only the positions beyond t0, stratified by the exact prefix
(proportional or Neyman allocation). This removes the early positions' variance exactly;
(b) *prefix tree with mass pruning*: expand nodes in order of probability and stop when the pruned mass bounds
the error. Deductive, with a rigorous error bound since p_illegal is in [0, 1];
(c) *state merging*: DP over (language state, model abstraction): RNN hidden states merged within epsilon (T352
style); hard-attention routing states; position-only models, whose layer-1 pre-activations are linear in the
one-hot tokens, so a DP over (stack state, quantised partial sums) is possible;
(d) structured importance sampling or control variates using the exact language law (e.g. a low-depth surrogate
model as a control variate).
Also report walks in embedding space (continuous relaxation) as a negative control only: the input law is discrete.
**Inputs.** MLP1 (built here first), `langs.Dyck2.enumerate_small` / `count`, `metrics.per_sequence`.
**Success (pre-registered).** Per architecture group, F <= 1 (strict) and F <= 10 (lenient) for Q_ill at one stated
budget in {1e3, 1e4, 1e5} FPE; separately state whether the method is deductive. Expected [speculative]: (a) beats
sampling at 1e4-1e5 FPE for every architecture, because the early positions carry most of the variance;
(b) and (c) are fully deductive only for rnn / hard / pos.
**Compute.** MLP1 build 4 CPU-h; truths at 1e8 samples for Q_ill (about 0.7 CPU-h per model, about 80 CPU-h:
spread over 2 containers or cut to 1e7 where sigma allows); estimator sweeps about 10 CPU-h.

## S3: rare bad outputs: estimate, locate, repair, rate

**Spec.** Target B_tau on dyck2 with tau per model, chosen so the truth is 1e-9 to 1e-5 (tau in {.5, .9, .99}; at
L = 32 most models have B0.9 < 1e-6). (i) Estimate: multilevel splitting over prefixes (score = running max
p_illegal), importance sampling that twists the language law toward long-range closers (deep stacks, distant
matching openers; cross-entropy-method fitted), and exact enumeration of the top-risk prefix families. Validate
exactly at L = 15, where `exact_small` gives B_tau exactly (dyck2 hard 2.9e-4, pos 1.1e-5, linear 1.5e-6).
(ii) Locate: gradient of a smoothed B_tau (sigmoid((max_t p_illegal - tau)/s)) with respect to the weights, weighted
by the rare-event estimator (as in H-R2). (iii) Repair: a targeted edit of <= 12 weights (gradient plus accuracy
Hessian, as in H-R3), with a random control of 40 same-size edits. (iv) Safe-use rating (as in T401):
r(k) = P(illegal output | the matching opener is >= k tokens back), computed from the weights.
**Inputs.** dyck2 baselines and MLP1; `metrics.per_sequence` (`max_ill`); `langs.Dyck2.annotate` (depth).
**Success (pre-registered).** (i) F_rel <= 1 strict and <= 10 lenient at one stated budget in {1e4, 1e5, 1e6} FPE on
>= 10 models with truths, and within 2x of the exact truth at L = 15 on >= 80% of models. (iii) An edit making
B_tau >= 10x rarer at a closer-accuracy cost <= .05 pp and Q_ill not worse, where <= 2 of 40 random controls do
as well. (iv) Rating within 2x of a conditional-MC check at >= 4 of 6 k values.
**Compute.** Estimator sweeps about 10 CPU-h; L = 32 rare truths by long MC (1e9 samples about 7 CPU-h per model)
for 10 models, about 70 CPU-h, or validated splitting; one or two containers.

## S4: certified upper bounds on the rare bad output

**Spec.** Rigorous upper bounds on B_tau and Q_ill for the exactness-friendly models (rnn, hard, relu, linear,
pos). Branch and bound over the prefix tree: a node is closed when a bound certifies that no completion of the
prefix can push p_illegal above tau. The bound over all continuations treats each future token as an element of
a finite set: IBP or CROWN / auto_LiRPA over the convex hull of the allowed embeddings. For hard attention, branch
on the routing; for pos, use the exact linear mixing; for relu / linear attention, use McCormick bounds on the
products. The bound = total probability of the open nodes plus exactly evaluated bad leaves. Validate soundness
against the exact L = 15 truths (a bound below the truth is a bug), then go to L = 32.
**Inputs.** dyck2 baselines and MLP1; auto_LiRPA (pip, CPU) or a hand-written IBP/CROWN for these small blocks.
**Success (pre-registered).** Soundness: 0 violations against the exact truths on every model at L = 15.
Usefulness at L = 32: certified UB <= the Clopper-Pearson 95% upper bound from plain sampling at the same charged
FPE (strict) or within 10x of it (lenient), on >= 50% of models in a group, at one stated budget in {1e5, 1e6}.
**Compute.** Branch and bound about 10-30 CPU-h; one container.

## S5: how far it scales (start after S2/S3 freeze a recipe; may build its model ladder immediately)

**Spec.** Rerun the frozen S2 / S3 / S4 winners on a ladder: vocabulary Dyck-k with k in {2, 4, 8}; length L in
{32, 64, 128}; layers {1, 2, 4}; width d {16, 32, 64}. Fit the cost law of exact or tree methods against sampling
(the crossover budget n* for each rung), and log compute estimates in docs/COMPUTE_FRONTIER.md wherever a rung
stops for compute.
**Inputs.** Frozen recipes from S2-S4; `Dyck2` generalised to k types (copy `langs.py` into the stream folder;
propose the change to the lead).
**Success (pre-registered).** A per-rung table of F against both bars; the stream succeeds if it predicts, from
the first two rungs, the F at the third within 2x on >= 2 of 3 axes.
**Compute.** Ladder of about 60 models x 5 min plus truths, about 20-40 CPU-h; two containers.

## Suggested order and dependencies

S1, S3 (L = 15 validation) and S4 (soundness at L = 15) can start at once on the baselines. S2 starts at once and
builds MLP1, which S3 and S4 then use for scoring. S5 follows the first frozen recipes. The lead is available to
review harness changes. Total about 150-250 CPU-h over 5-7 containers, no GPU needed.
