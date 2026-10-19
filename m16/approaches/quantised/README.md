# quantised: T352, quantised versions of M_16,10 (QAT retrains, discrete inputs, state census, exact DP)

**Question.** If the hidden activations (and optionally the inputs) of the M_16,10 architecture are quantised, how much
accuracy is lost against float, how many hidden states are actually reachable, and does an exact forward dynamic
program over (state, input interval, label summary) become cheaper than sampling at equal FLOPs?

**Status.** PRE-REGISTERED 2026-10-18 (this section was written and pushed before any training or measurement).
Owner: Claude session_016BGedpds19gq95yjWmDMn5 (T192 child), DRI session_01CD3g9Cvc3qk3pM11eyXXzj. Budget cap 20
Colab units (stop and report at the cap; runtimes stopped when idle). No Fable.

## Pre-registration (fixed before any result)

**Architecture.** H = 16, T = 10, h_0 = 0, no biases, h_t = q(Wh h_{t-1} + Wi x_t), logits = Wo h_10, pick =
argmax(logits); label = position of the 2nd-largest input. Weights stay float32 in training and float64 in evaluation;
only activations (and in (2) inputs) are quantised. q variants:
- **float**: q = relu (control, same trainer).
- **binary**: q(u) = 1[u > 0] in {0, 1}; straight-through gradient dq/du = 1[|u| <= 1] (hard-tanh STE).
- **int4 / int8**: q(u) = s * clip(round(u / s), 0, L - 1) with L = 16 / 256 levels; s > 0 a learnable scalar per
  model (LSQ: d q/d u = 1 inside [0, (L-1) s], 0 outside; d q/d s = round(u/s) - u/s inside, L - 1 above, 0 below,
  gradient scaled by 1/sqrt(B * H * (L - 1))), initialised from the first batch as s0 = 2 mean(relu(u)) / sqrt(L - 1).

**Training recipe = M_16,10's zoo recipe** (`../capacity_ceiling/trainer.py`): zoo init per seed, AdamW lr 0.04/sqrt(16),
wd 1e-3, betas (0.9, 0.999), batch 1024, 2^30 training inputs (2^20 steps), warm-up 2^14 steps, exponential decay to
0.01x at mid-run; cross-entropy on the 2nd-argmax label. Fresh N(0, I) data per step.

**(1) QAT, continuous inputs.** float, binary, int4, int8; seeds 0-4 (5 each). Plus, as a free extra, post-training
quantisation (PTQ) of the actual M_16,10 weights (binary, int4, int8 with s calibrated on 1e6 inputs, no retraining).

**(2) Discrete inputs.** x_t replaced by its nearest level on a uniform grid of L_x levels on [-3, 3] (int4: 16 levels,
step 0.4; int8: 256 levels, step 6/255); tails clipped into the end levels; level masses = Gaussian mass of each level's
Voronoi cell. Label on the discretised inputs, ties broken by the LOWER index (stable descending order). Crossed with
binary and int4 activations: 4 cells x seeds 0-2 (3 each).

**Accuracy.** >= 1e8 fresh inputs per model, float64 forward, independent CUDA Philox stream per model; compared with the
float control (and with M_16,10's .9531175). Reported per variant: mean, sd, min, max over seeds.

**(3) State census (T348-style).** >= 1e9 inputs per variant (first seed of each), GPU, distinct hidden states h_t per
step t = 1..10 (64-bit hash of the quantised state; exact key for binary), plus singletons and exp(H). For binary
activations also the EXACT reachable set per step by forward closure (all input intervals from every reachable state).

**(4) Exact forward DP.** For a variant with discrete state space and continuous inputs:
P(correct) = sum_p int phi(v) P[x_p = v, exactly one other x_q > v, pick(h_10) = p] dv. For fixed (p, v) a DP over
(state, c) with c = number of earlier/later inputs above v in {0, 1, >= 2}: at t != p the input line is cut at the
state's activation breakpoints and at v, each piece has an exact Gaussian mass (erfc, float64) and a fixed next state;
at t = p the input is v. The v-integral is done by Gauss-Legendre on panels in u = Phi(v) with refinement (error
reported as the change on doubling the nodes). For discrete-input variants: the same DP with the v-integral replaced
by the exact finite sum over levels (ties by lower index, so "beats p" = x_q > v or (x_q = v and q < p)) =
exact enumeration. Run for every variant whose reachable state count makes it feasible (binary first; int4 only if
its reachable count allows; otherwise a scoped negative with the counts).
**Cost comparison.** DP FLOPs counted with the T196 convention (multiply-add = 2; erfc / sort charged at the counted
calls); sampling cost = the quantised model's own forward FLOPs per input (2 * (16 * 16 + 16) * 10 + 2 * 10 * 16 plus
the quantisation ops). At equal FLOPs, n = F_DP / F_forward samples, sampling MSE = p (1 - p) / n vs the DP's squared
error (refinement estimate; DP also checked against a >= 1e9-sample MC of the same model). Ratio reported per variant.

**Amendment 1 (2026-10-19 00:05 UTC, before any final result; pushed before the runs it governs).** In the main run the
binary (hard-tanh |u| <= 1 STE) models sit at train accuracy ~.30 after 98k of 1M steps while float/int4/int8 reach
.72-.90 (likely vanishing STE gradients: binary states make |u| > 1 common). Per the alternatives above, all binary cells
(5 continuous-input + 6 discrete-input models, same seeds) are ALSO trained with two fallback STEs: hard-tanh |u| <= 3
(`--ste ht3`) and tanh' (`--ste tanh`). All three are reported. Census and DP use, per binary cell, the STE variant
with the highest mean float64 accuracy (chosen on accuracy only, before census/DP).

**Amendment 2 (2026-10-19 04:00 UTC, DRI request; pushed before the runs it governs).** Width ladder for the owner's
question (does a quantised net that is still ~95% accurate become exactly solvable?): binary (ht1 STE, the amendment-1
choice) and int4 (LSQ) activations at H = 32 and H = 64, seeds 0-2, continuous inputs, otherwise the same zoo recipe
(lr 0.04/sqrt(H), zoo init of DistRNN(H)), 1e8 float64 evaluation. Reported: the first width reaching >= .95 mean
accuracy per activation type (or none), and its reachable-state census (1e9 inputs; binary keys exact up to 64 bits,
int4 hashed). Skipped if the projected spend would pass the 20-unit cap. Also (DRI): tie rate and tie-free accuracy for
the discrete-input cells (label convention in training and evaluation: 2nd-largest of the DISCRETISED inputs, ties
broken by the lower index).

**Deviation 1 (2026-10-19 05:55 UTC, before these census runs).** The census of int8 (continuous inputs), int4 x int4
inputs and int4 x int8 inputs uses 2.5e8 inputs (not >= 1e9): at 1e9 the int8 run ran out of T4 memory with 32 partitions
(nearly every key distinct), more partitions would cost ~3 h (> 3 units), and the int4 census at 1e9 already shows
864,652,487 distinct states at step 10 (86% of inputs), i.e. these high-precision variants have effectively
per-input states; 2.5e8 suffices to show that. Binary (3 cells) and int4 (continuous) used 1e9.

**Done when** each variant has accuracy, state counts and DP cost vs sampling, or a scoped negative saying which variant
breaks and why. Heavy CPU slot requested from the DRI only if the DP needs one.

**Alternatives to keep in mind (written before results).** Accuracy loss may come from the STE/optimiser rather than
from capacity (check: train curves, a second STE for binary if it fails to train); a small reachable-state count may be
an artefact of dead neurons (report per-neuron activity); DP cost may be dominated by the number of breakpoint
intervals, not states.

## Method
`qtrain.py` (stacked QAT trainer, CUDA graph, resumable; T4), `qeval.py` (float64 1e8-sample accuracy, PTQ),
`qcensus.py` (state census), `qdp.py` (binary exact closure + DP). Runs: `q_main.sh` (32 models, seeds 0-4 / 0-2,
data seed 352), `q_ste.sh` (amendment 1). Checkpoints in data/quantised/ (not in git).

## Results ([measured], unverified)
**(1)/(2) Accuracy** (1e8 float64 inputs per model, independent stream; mean / sd / min / max over seeds; chance .10;
M_16,10 float .95311 with the same evaluator; trained float control below uses the same trainer and seeds):

| variant | seeds | mean | sd | min | max |
|---|---|---|---|---|---|
| float (relu) | 5 | 0.9061 | 0.0315 | 0.8816 | 0.9541 |
| binary ht1 (pre-reg) | 5 | 0.2990 | 0.0202 | 0.2732 | 0.3229 |
| binary ht3 | 5 | 0.2019 | 0.0123 | 0.1898 | 0.2186 |
| binary tanh' | 5 | 0.2968 | 0.0280 | 0.2509 | 0.3177 |
| int4 LSQ | 5 | 0.7236 | 0.0065 | 0.7166 | 0.7320 |
| int8 LSQ | 5 | 0.8989 | 0.0585 | 0.7966 | 0.9407 |
| binary ht1, x int4 | 3 | 0.2801 | 0.0379 | 0.2365 | 0.3052 |
| binary ht3, x int4 | 3 | 0.2075 | 0.0184 | 0.1869 | 0.2222 |
| binary tanh', x int4 | 3 | 0.2669 | 0.0391 | 0.2344 | 0.3103 |
| int4, x int4 | 3 | 0.8398 | 0.0125 | 0.8261 | 0.8506 |
| binary ht1, x int8 | 3 | 0.2924 | 0.0262 | 0.2631 | 0.3135 |
| binary ht3, x int8 | 3 | 0.1698 | 0.0495 | 0.1132 | 0.2051 |
| binary tanh', x int8 | 3 | 0.2807 | 0.0284 | 0.2537 | 0.3103 |
| int4, x int8 | 3 | 0.7438 | 0.0049 | 0.7382 | 0.7474 |

PTQ of M_16,10 itself (no retraining; s = 99.99th pct of positive pre-activations / (L-1)): float .95311, int8 .92784,
int4 .34192, binary .16305.
Binary STE choice (amendment 1, by mean accuracy only): the pre-registered hard-tanh |u| <= 1 (ht1) is best in all
three binary cells (continuous .2990 vs tanh' .2968 vs ht3 .2019; x int4 .2801 / .2669 / .2075; x int8 .2924 / .2807 /
.1698); census and DP use ht1.

**Ties (discrete inputs)** (`python qties.py qat.pt 1e7`, `ties.json`; same convention in training and evaluation:
label = 2nd-largest position of the discretised input, ties broken by the lower index; "tied" = v1 == v2 or v2 == v3):
tie rate .581 on the int4 input grid (step 0.4) and .047 on the int8 grid (step .0235). Accuracy all / tie-free / tied:
int4 act x int4 inputs .850/.878/.831, .826/.850/.809, .842/.877/.817 (seeds 0-2); int4 x int8 inputs .747/.763/.428,
.746/.762/.406, .738/.754/.426; binary x int4 .237/.269/.213, .305/.345/.276, .299/.339/.269; binary x int8
.263/.265/.224, .301/.302/.262, .315/.315/.278. Reading [inferred]: the int4 input grid is easier, not because ties are
scored leniently (tied inputs score below tie-free ones), but because a 0.4 grid turns every near-tie (gap < 0.4) into
an exact tie decided by a fixed index rule, which the net learns (.81-.83 on tied inputs), and leaves only tie-free
inputs with top-3 gaps >= 0.4, which are easy (.85-.88). The continuous task's hard near-tie cases no longer exist.

**(4) Exact DP, binary activations** (`qdp.py`; ht1 models; exact reachable sets by closure: continuous inputs 4,487 -
11,966 states of 65,536 per model, discrete int4 inputs 1,119 - 1,977, int8 inputs 2,672 - 8,780; <= 17 input
intervals per state). Discrete inputs: exact enumeration (float64). Continuous inputs: the v-integral on M uniform
Gauss-Legendre panels (n = 8); the integrand jumps at every breakpoint, so error falls ~1/M; the exact breakpoint-panel
mode (degree-9 polynomial per panel, 5 nodes) would need ~127k panels x 5 nodes = ~1e14 FLOPs per model (not run).
Comparison with sampling at EQUAL FLOPs (binary forward cost 5.95e3 FLOPs per input; se_eq = sqrt(p(1-p)/n) with
n = DP FLOPs / 5.95e3):

| model | DP mode | DP P(correct) | DP FLOPs | MC 1e8 (acc +- se) | z(DP - MC) | MC se at equal FLOPs |
|---|---|---|---|---|---|---|
| bin_s0 | M=16,n=8 | 0.3000170 | 2.49e+10 | 0.3002312 +- 4.6e-05 | -4.7 | 2.2e-04 |
| bin_s0 | M=32,n=8 | 0.3001543 | 4.99e+10 | 0.3002312 +- 4.6e-05 | -1.7 | 1.6e-04 |
| bin_s0 | M=64,n=8 | 0.3002246 | 9.97e+10 | 0.3002312 +- 4.6e-05 | -0.1 | 1.1e-04 |
| bin_x16_s0 | exact enum | 0.2365765 | 2.80e+08 | 0.2365329 +- 4.2e-05 | +1.0 | 2.0e-03 |
| bin_x16_s1 | exact enum | 0.3053044 | 2.55e+08 | 0.3052475 +- 4.6e-05 | +1.2 | 2.2e-03 |
| bin_x16_s2 | exact enum | 0.2985082 | 1.68e+08 | 0.2984588 +- 4.6e-05 | +1.1 | 2.7e-03 |
| bin_x256_s0 | exact enum | 0.2630290 | 9.67e+10 | 0.2630698 +- 4.4e-05 | -0.9 | 1.1e-04 |
| bin_x256_s1 | exact enum | 0.3006625 | 1.67e+11 | 0.3007091 +- 4.6e-05 | -1.0 | 8.6e-05 |
| bin_x256_s2 | exact enum | 0.3134904 | 3.34e+11 | 0.3134862 +- 4.6e-05 | +0.1 | 6.2e-05 |

Reading [measured/inferred]: on a discrete input grid the exact DP is far cheaper than sampling (x16: exact at 1.7-2.8e8
FLOPs, where sampling at equal cost has se ~2e-3; x256: exact at 1e11-3e11 FLOPs vs sampling se ~1e-4). With continuous
inputs the uniform-panel DP at M = 64 (1e11 FLOPs) has error ~7e-5 (Richardson estimate from M = 32/64; z vs MC +-0)
vs sampling se 1.1e-4 at equal FLOPs: about break-even; exactness needs breakpoint panels (~1e14 FLOPs).

**(3) Census of reachable hidden states** (`qcensus.py CKPT NAME N P`, `census_<name>.json`; x ~ N(0, I) float32
CUDA Philox seed 3521000 + model index, discretised for grid cells; distinct state codes per step by sort + unique in
P hash partitions; keys exact for binary and int4 H = 16, hashed to 64 bits otherwise, with a second mixer at step 10
as a collision check: equal counts in every run). Distinct states at steps 1 / 2 / 3 / 5 / 10, singletons and exp(H)
at step 10:

| model (seed 0) | N | t=1 | t=2 | t=3 | t=5 | t=10 | singletons t=10 | exp(H) t=10 |
|---|---|---|---|---|---|---|---|---|
| binary ht1 | 1e9 | 3 | 12 | 51 | 335 | 3,755 | 102 | 566 |
| binary ht1, x int4 | 1e9 | 2 | 13 | 53 | 173 | 1,734 | 70 | 64.5 |
| binary ht1, x int8 | 1e9 | 2 | 9 | 22 | 76 | 2,241 | 108 | 138 |
| int4 | 1e9 | 135 | 8,410 | 343,297 | 84,905,953 | 864,652,487 | 807,341,446 | 7.28e8 |
| int4, x int4 | 2.5e8 | 16 | 200 | 2,628 | 309,519 | 126,757,342 | 90,669,587 | 7.15e7 |
| int4, x int8 | 2.5e8 | 80 | 5,066 | 292,966 | 64,822,143 | 241,483,012 | 235,126,678 | 2.37e8 |
| int8 | 2.5e8 | 935 | 484,535 | 54,345,744 | 249,956,530 | 250,000,000 | 250,000,000 | 2.5e8 |
| int4, H = 64 | 2.5e8 | H64_T1 | H64_T2 | H64_T3 | H64_T5 | H64_T10 | H64_F1 | H64_EH |

Sampled binary counts (3,755 continuous) are below the exact closures (4,487-11,966) because many reachable states
have tiny mass. int4 / int8 states at step 10 are close to one per input.

**Width ladder (amendment 2)** (`q_h32.sh`, `q_h64.sh`; `qat_h32_eval.json`, `qat_h64_eval.json`; 1e8 float64 inputs
per model, seeds 0-2):

| H | binary ht1 | int4 LSQ |
|---|---|---|
| 16 | .299 (5 seeds, above) | .724 (5 seeds, above) |
| 32 | .3631 / .3419 / .3906 | .9020 / .8868 / .9029 |
| 64 | .4786 / .5212 / .4319 | .9515 / .9552 / .9548 (mean .9538) |

First width with mean accuracy >= .95: int4 at H = 64; binary none up to H = 64. The H = 64 int4 census (row above,
2.5e8 inputs per Deviation 2) gives its reachable states.

**Deviation 2 (2026-10-19 07:05 UTC, after the 1e9 run failed, before its rerun).** The int4 H = 64 census at 1e9
inputs ran out of T4 memory (64 partitions, 11 step buffers of near-unique hashed keys); it was rerun at 2.5e8, the same
reduction as Deviation 1 for the other high-cardinality variants.

## Compute accounting
- Colab: T4 runtimes only (~1.07 units/h). Balance 164.37 before T352, 151.69 at 07:00 UTC 2026-10-19 with one T4
  still running the last census; final spend UNITS_FINAL units of the 20-unit cap. Runs: main QAT grid (32 models,
  2^30 inputs each, stacked) ~2.7 h; STE fallbacks ht3 / tanh' ~1.3 h each; width ladder H = 32 and H = 64 ~1.8 h each
  (in parallel on one T4); 1e8 float64 evaluations of every model; census 7 + 1 runs (72 s - 46 min each); the rest is
  idle keepalive and setup.
- Model FLOPs: training 2^30 inputs x (forward + backward ~3x forward) per model; forward ~2 x 10 x (16 x 16 + 16)
  FLOPs per input at H = 16. Evaluation 1e8 inputs per model. Census 1e9 or 2.5e8 forwards per model.
- DP: FLOP counts in the DP table (closure + DP), counted as multiply-adds x 2, CPU (local, <= 2 GB).

## Claims
- [measured] At H = 16 with the zoo recipe: QAT accuracy float .906, int8 .899, int4 .724, binary .30 (best of three
  STEs); PTQ of M_16,10 keeps int8 (.928) but not int4 (.342) or binary (.163).
- [measured] Binary nets have few reachable states (closure 4.5k-12k continuous, 1.1k-8.8k discrete), and their exact
  DP is exact on discrete inputs at 1.7e8-3.3e11 FLOPs, agreeing with 1e8-sample MC within 1.2 se.
- [measured] int4 and int8 nets have close to one state per input at step 10 (86% distinct at 1e9 for int4, 100% at
  2.5e8 for int8).
- [measured] The width ladder reaches >= .95 only for int4 at H = 64; binary stays at or below .52 up to H = 64.
- [inferred] No quantised variant tested is both >= .95 accurate and exactly solvable by state enumeration: the
  accurate ones have per-input states, and the solvable ones (binary) are inaccurate.
- [inferred] On discrete inputs the exact DP beats sampling at equal FLOPs by orders of magnitude (x16 grid); on
  continuous inputs the uniform-panel DP is only break-even with sampling at M = 64.

## Alternatives not ruled out
- Binary accuracy may be limited by the training recipe (STE, lr, schedule, 2^30 inputs, initialisation) rather
  than by capacity. Three STEs agree, but no recipe search, longer training or distillation from M_16,10 was tried.
- int4 at H = 16 might reach higher accuracy with per-neuron scales, a learned zero point or longer training.
- The int4 H = 64 net may have a small "effective" state set under a coarser equivalence (states that give the same
  output for all futures). The census counts raw codes, which is an upper bound on the DP state count after such a
  minimisation. Minimisation was not tried.
- The census uses float32 forwards; rare float32/float64 threshold disagreements are ignored (they affect a few codes,
  not the order of magnitude).

## Failure modes
- The census needs memory proportional to the number of distinct keys per partition; near-unique variants needed
  input reduction (Deviations 1-2), so their counts are lower bounds on the count at 1e9.
- The continuous-input DP is not exact with uniform panels (error ~1/M); exact breakpoint panels cost ~1e14 FLOPs.
- Hashed keys (int4 H > 16, int8) can collide; the second mixer agreed in every run, but that is evidence, not proof.
- Discrete-input results depend on the tie convention (lower index); the int4 grid has a 58% tie rate.
