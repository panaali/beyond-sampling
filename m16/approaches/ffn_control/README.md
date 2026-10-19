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

### Amendment A (2026-10-18 ~23:55 UTC, before any ladder run): owner retarget via the DRI

The owner changed the target. The goal is now **the simplest all-inputs-at-once network that reaches >= .95
accuracy** (M_16,10: .9531): fewest layers first, then fewest parameters. Then attempt the challenge on it: a
deductive accuracy estimate whose MSE is no worse than sampling at equal FLOPs (F <= 1). The 12 runs above continue
as cheap ladder points and as the original 432-parameter control.

Step 1, simplicity ladder (bias-free ReLU MLPs, the same task, recipe and label as above; equal widths per layer):
- Depth 1: widths 16, 32, 64, 128, 256, 512. Depth 2: widths 16, 24, 32, 64. Depth 3: widths 16, 24, 32.
  Names `D<depth>w<width>` (`python train.py D2w32 SEED 28`).
- Screen: seed 0 of every configuration at n_train = 2^28 (a quarter of the zoo's 2^30, for CPU). Seeds 1 and 2
  (>= 3 seeds) for every configuration whose seed 0 reaches >= .94, and for the narrowest configuration of each
  depth whose seed 0 reaches >= .95. The chosen configuration is retrained at 2^30 on 3 seeds to check
  that the shorter schedule did not decide the outcome. If no depth-1 width up to 512 reaches .95, depth 1 is
  reported as plateaued at the best width, with its accuracy curve.
- Accuracy on 1e7 fresh inputs (seed 20261018) for each (depth, width, params, seed), with seed spread.
- Region counts per configuration. Depth 1, bias-free: the regions are the chambers of a central arrangement of w
  hyperplanes in R^10, exactly 2·sum_{i<10} C(w-1, i) in general position (5.6e4 at w = 16, 1.4e6 at w = 22, 6.3e7 at w = 32,
  5.6e10 at w = 64, 3.8e13 at w = 128, 2.3e16 at w = 256). Deeper nets: this bound for layer 1, plus sample-based distinct gate
  patterns per layer and mass-coverage counts N_delta from 1e7 inputs (`analyze.py regions`).
- Selection: (a) simplest by depth, then parameters, reaching >= .95 on all seeds; (b) most tractable reaching
  >= .95 on all seeds, judged by realised region / decision-cell counts and N_delta (lowest wins). Both reported.
- Optional comparison only: a permutation-structured variant, if cheap.
- Added rungs (2026-10-19 ~00:40, after seeing seed-0 screens D2w32 .855, D3w24 .863, D3w32 .961, D2w64 .93 at 3/4 of
  training): D2w128 and D3w28 (seed 0), to bracket the narrowest depth-2 and depth-3 widths. Seeds 1-2 of D3w32
  started per the rule above.
- Added rungs (2026-10-19 ~01:30): D3w32 seed 1 reached only .926 (seed 0 .961), D2w64 seed 0 .932, so D3w48, D3w64
  and D4w32 (seed 0) are added.
- Added (2026-10-19 ~02:10): D1w128 seed 0 reached .965 (D1w64 .606), so D1w128 seeds 1-2 and a D1w96 rung are added.
- Added (2026-10-19 ~03:10): the pre-registered 2^30 retrain of D1w128 seed 0 reached .990 (vs .965 at 2^28), so the
  schedule matters. 2^30 runs of D1w96 (seeds 0-2) and D1w64 (seed 0) are added to locate the depth-1 threshold
  under the zoo's full schedule.
- Added (2026-10-19 ~04:00): at 2^30 D1w128 reaches .990 on all 3 seeds, D1w96 seed 0 .974, D1w64 .602, so a D1w80
  rung (2^30, seed 0) is added.

Step 2, on the chosen network(s): the pre-registered stage-2/3 analysis above (region counts vs the RNN's x23-33
per step, exact-accuracy attempt, exact walks, F at C >= 1k FPE vs plain sampling, M_16,10 control with the same
code), on 3 seeds plus 9 noised copies (sigma .05). Additional candidate methods named by the DRI: exact Gaussian
structure of the first-layer pre-activations (jointly Gaussian with covariance W1 W1^T) and approximate
permutation symmetry. A short mechanistic account of what the neurons compute is part of done.

Cost accounting for walks (added before the scored panel): F is reported under three charges: `F` = the
implemented algorithm (a full forward-with-slope per piece, 2 FPE), `F_incr` = an incremental cost model
(after a gate flip, a rank-1 update plus recompute of only the later layers / steps; a model, not implemented),
`F_floor` = 4T FLOPs per piece (a lower bound for any piece walk). `F` is the headline.

### Amendment B (2026-10-19 ~00:10 UTC, before any scored panel result): the scoring panel

The pre-registered panel (9 copies with noise sigma .05 x RMS per matrix) is replaced. At that sigma M_16,10's
copies collapse to accuracy .26-.45 (measured with `m16/arc_eval.py`'s own `noisy_models`, 6 copies), far from the
regime the project scores in (sealed retrain panels; T337 breakdown already at relative .0025). The only panel
model scored before the change was M_16,10 itself and one copy (log kept in data/, not used). New panel
(`analyze.panel`), with relative noise sigma = .001:
- each MLP configuration: its trained seeds plus 3 noised copies per seed (12 models for 3 seeds);
- M_16,10 control: the target, alg_zoo retrains `zoo_2nd_argmax(16, 10, seed=0..4)` (accuracies .88-.95), and 6
  noised copies of the target (12 models).

### T387 pre-registration (2026-10-19 ~02:50 UTC, before any fixed-set scoring): fixed great-circle sets

DRI request: make D1w128's estimator deterministic (no random model inputs), QMC-class like T10, by replacing the
random great circles of `fast1.py` with FIXED sets, identical for every model, fixed before scoring.
- Panel: the same 12 D1w128 models (3 seeds + 9 copies at relative sigma .001). Truth: 1e8-sample MC per model
  (seed 8787, disjoint from every other seed here); its noise is reported next to each F.
- Budget: C = 4,000 FPE. Each set has n circles, n chosen once from the mean walk cost measured before this
  pre-registration (17.2 FPE): n = 232. Cost charged as executed (sum of `fast1` FLOPs / FPE), C_eff = max(C, 1k).
- Map from the unit cube to circles: point q in [0,1)^20 -> (u, v) = (Phi^-1(q_1..10), Phi^-1(q_11..20)).
- Designs (all fixed now; nothing is tuned on truths):
  - S: scrambled Sobol (scipy.stats.qmc.Sobol, d = 20, scramble=True), first 232 points, seeds 1-5 (5 sets).
  - H: scrambled Halton (scipy.stats.qmc.Halton, d = 20), first 232 points, seeds 1-5 (5 sets).
  - K: rank-1 Korobov lattice, n = 233 (prime; one more circle), generator z = (1, a, a^2, ...) mod 233,
    points frac((i z + 0.5) / 233), a in {17, 31, 57, 89, 113} (5 sets, no search).
  - P (symmetry-derived): 23 Sobol points (seed 1-5) x the 10 cyclic shifts of the input coordinates applied to
    both u and v (230 circles; 5 sets). This uses the permutation symmetry of the input law, not W1. A design
    derived from W1's comparator structure is attempted only if one can be stated without looking at truths;
    if none is found, that is reported as not done.
  - Control R: the random circles of `score_fast1.py` (F .280 [.253, .306], direct .267).
- Report, per design family: F per set with a bootstrap CI over the 12 models, the mean and spread of F across
  the 5 sets, and the RMS quadrature error. A family "keeps F < 1" only if all 5 sets do.
- Classification: deterministic quadrature over (u, v), i.e. QMC-class under project rules (like T10), not
  deductive in ARC's sense.

## Method
Key files:
- `train.py`: bias-free ReLU MLP training with alg_zoo's recipe (`D<depth>w<width>` ladder names; `L1h22` etc. controls).
- `models/*.json`: trained weights plus loss/accuracy logs (`_n28` = trained on 2^28 samples, none = 2^30).
- `analyze.py`: accuracy (`acc`), region statistics (`regions`), the scoring panel (`panel`: seeds + noised copies).
- `walks.py`: generic exact 1-D walks for any bias-free ReLU MLP and for the M_16,10 RNN, with FLOP accounting
  (coordinate lines with Gaussian weight; great circles as two lines with Cauchy weight). Reproduces
  `m16/lineint.py` exactly on M_16,10 (5/5 test lines, same values and piece counts).
- `fast1.py`: an incremental, vectorised exact great-circle walk for depth-1 nets. Every hidden unit flips exactly
  twice on a great circle at angles known in closed form, each flip is a rank-1 update of the logit coefficients,
  and the correct set on each sub-arc is one interval. Matches `walks.circle_walk` to 6e-9.
- `score.py`, `score_fast1.py`: F on a panel, F = mean_m(Var_m c_m) / mean_m p_m(1-p_m) (budget-independent for an
  average of iid walks at C >= 1k FPE), bootstrap CI over models; `score_fast1.py` also runs a direct budget check
  (40 estimates per model at C = 4000 FPE against an independent 1e7-sample truth).
- `cone_test.py`: the DRI's rank-order-cone idea for depth 1 (comparator exactness, equivariance, subregions/cone).

All walk estimators here are **sampling-class**: the lines and great circles are random, and only the integral
along each is exact. None is deductive in ARC's sense.

## Status
Step 1 done (ladder; 2^30 retrain of the chosen D1w128 and the 9 remaining 432-parameter control seeds still
running). Step 2: F reported for every candidate and for M_16,10; a deductive estimator was not achieved
(scoped negatives below).

## Results

### Step 1: simplicity ladder (bias-free ReLU MLPs, 2^28 samples, accuracy on 1e6 fresh inputs; 1e7 table: `results/acc.json`)

| depth | width | params | seed 0 | seed 1 | seed 2 | >= .95 on all seeds |
|---|---|---|---|---|---|---|
| 1 | 16 / 32 | 320 / 640 | see acc.json | | | no |
| 1 | 64 | 1,280 | .606 | | | no |
| 1 | 96 | 1,920 | .790 | | | no |
| 1 | **128** | **2,560** | **.965** | **.954** | **.965** | **yes (simplest)** |
| 2 | 24 | 1,056 | .825 | | | no |
| 2 | 32 | 1,664 | .855 | | | no |
| 2 | 64 | 5,376 | .932 | | | no |
| 2 | 128 | 18,944 | .980 | .973 | .966 | yes |
| 3 | 24 | 1,632 | .863 | | | no |
| 3 | 28 | 2,128 | .968 | .884 | (acc.json) | no |
| 3 | 32 | 2,688 | .961 | .926 | .968 | no |
| 3 | 48 | 5,568 | .981 | (acc.json) | | |
| 3 | 64 | 9,472 | .986 | .985 | .980 | yes |
| 4 | 32 | 3,712 | .981 | .976 | .986 | yes (most tractable by regions) |
| controls (2^30) | L1h22 / L1h43 / L2h13 | 440 / 860 / 429 | .358 / .507 / .690 | | | no |

2^30 samples (the zoo's full schedule; accuracy on 1e6 fresh inputs):

| depth | width | params | seed 0 | seed 1 | seed 2 | >= .95 on all seeds |
|---|---|---|---|---|---|---|
| 1 | 64 | 1,280 | .602 | | | no |
| 1 | 80 | 1,600 | .659 | | | no |
| 1 | 96 | 1,920 | .974 | .904 | .973 | no |
| 1 | **128** | **2,560** | **.990** | **.990** | **.990** | **yes** |

M_16,10: .9531 with 432 parameters. Reproduce: `python train.py D1w128 0 28` (or `... 30`), `python analyze.py acc`.

- **Simplest by depth, then parameters:** D1w128, one hidden layer of 128, 2,560 parameters (5.9x M_16,10), under
  both schedules (at 2^30 D1w96 passes on 2 of 3 seeds only). The longer schedule matters (.965 -> .990 for D1w128;
  D1w96 .79 -> .97), so deeper configs that failed at 2^28 might pass at 2^30; only depth 1 was retrained.
  Depth 1 jumps from .79 at width 96 to .95-.965 at 128. No depth-2 or depth-3 net with fewer parameters passes on
  all seeds (D3w28/D3w32 pass on some seeds only: SGD lands in worse minima on others).
- **Most tractable by the pre-registered region criterion:** D4w32 (lowest realised counts, below).
- At the 432-parameter budget no MLP comes close: .36 (1 layer), .69 (2 layers) vs the RNN's .953. Weight reuse
  over 10 steps buys the RNN roughly 6x in parameters on this task.

### Step 1: region counts (`python analyze.py regions NAMES`; 1e7 inputs, seed 20261019)

Distinct gate patterns seen in 1e7 samples (a lower bound on realised regions), per layer, and the share of the
last layer's samples whose region was seen once (Good-Turing estimate of the unseen mass). Layer-1 bound =
2·sum_{i<10} C(w-1, i) (central arrangement in general position).

| net | layer-1 bound | distinct per layer (1e7 samples) | seen once (last layer) |
|---|---|---|---|
| D1w128 | 3.8e13 | 9.93M | 98.6% |
| D2w128 | 3.8e13 | 9.84M, 10.0M | ~100% |
| D3w64 | 5.6e10 | 5.8M, 9.8M, 9.9M | 97.9% |
| D3w32 (3 seeds) | 6.3e7 | 0.48-0.64M, 3.3-4.5M, 4.5-5.3M | 30-37% |
| D4w32 | 6.3e7 | 0.44M, 3.0M, 4.6M, 5.5M | 39.4% |
| M_16,10, per step 1..10 | | 2, 34, 689, 7.4k, 46k, 190k, 557k, 1.2M, 2.1M, 3.2M | 23% |

(M_16,10's exact prefix counts are 34/792/21,032/632,394 at k = 2..5, H-E13; the sample counts above undercount
them from k = 3 on.) The N_delta coverage counts of the pre-registration are not estimable from 1e7 samples for
any of these nets: the tail mass sits in singletons, so every N_delta for delta <= 1e-2 just equals "distinct minus
the singleton tail" (values in `results/regions_*.json`, not meaningful).

**Region counts do not favour the MLPs.** Every MLP that reaches .95 is at least as fragmented as the 10-step RNN
in this sampled sense, and the simplest one (D1w128) puts almost every input in its own region. Exact region
enumeration with exact masses is therefore dead for these MLPs too, at any budget where comparison with sampling
is meaningful (scoped: region-by-region methods).

### Step 2: exact walks vs plain sampling (F; plain sampling = 1)

Panels of 12 models: each MLP config's 3 seeds plus 3 noised copies per seed (relative sigma .001); M_16,10's
target, zoo retrains 0-4 and 6 noised copies. F with a 95% bootstrap CI over models. FPE = the network's own
forward FLOPs (M_16,10 5,920; D1w128 5,248; D3w32 5,472; D4w32 7,552; D3w64 19,136).

| net | estimator | F (implemented) | F_incr (cost model) | F_floor | pieces / circle | var. reduction |
|---|---|---|---|---|---|---|
| M_16,10 | coordinate walks | 31.0 [27.3, 35.2] | 11.0 | 1.48 | | |
| M_16,10 | great circles | 15.4 [12.5, 18.3] | 6.2 | 0.29 | 221 | 29x |
| D3w32 | great circles | 5.7 [4.7, 6.3] | 1.9 | 0.15 | 117 | 50x |
| D3w64 | great circles | 5.9 [4.5, 7.6] | 1.7 | 0.05 | 264 | 110x |
| D4w32 | great circles | 3.8 [3.0, 4.5] | 1.6 | 0.07 | 135 | 87x |
| D1w128 | great circles (generic) | 8.7 [7.2, 10.2] | 0.20 [0.16, 0.23] | 0.16 | 256 | 64x |
| **D1w128** | **great circles, `fast1.py`** | **0.280 [0.253, 0.306]** | | | 256 | 64x |
| D1w128 | `fast1.py`, direct at C = 4k FPE | **0.267 [0.245, 0.291]** | | | | |
| D1w128@30 (2^30 retrains, acc .990) | `fast1.py` | **0.104 [0.091, 0.115]**; direct .103 | | | 256 | |
| D1w96 (1 seed + 3 copies) | `fast1.py` | 0.87 [0.84, 0.90] | | | | |
| D1w64 (1 seed + 3 copies) | `fast1.py` | 1.08 [1.04, 1.12] | | | | |
| L1h43 (1 seed + 3 copies, acc .51) | `fast1.py` | 1.19 [1.17, 1.22] | | | | |
| L1h22 (1 seed + 3 copies, acc .36) | `fast1.py` | 2.32 [2.27, 2.36] | | | | |

Reproduce: `python score.py M16_10 3000 600`, `python score.py D3w32 1000 400`, `python score.py D3w64 600 200`,
`python score.py D4w32 600 300`, `python score.py D1w128 300 200`, `python score_fast1.py D1w128 3000 40`,
`python score_fast1.py D1w96 2000 20`. Walk means agree with independent 1e7-sample MC on every model (|z| < 2).

- **Headline (sampling-class, not ARC-valid):** on the simplest .95 MLP, exact great-circle walks beat plain
  sampling at equal FLOPs by 3.6x in MSE (F .28, confirmed by a direct budget check, .27). The same family of
  walks loses to sampling on M_16,10 by 15x as implemented (6x under the incremental cost model).
- **Why depth 1 wins:** a great circle crosses each first-layer hyperplane exactly twice, at angles known from W1
  in closed form, and each crossing changes the logits by a rank-1 term (40 FLOPs). Deeper nets and the RNN have
  to propagate each flip through every later layer / step (F_incr 1.6-6.2), and the implemented generic walk pays
  a full forward-with-slope per piece. The incremental model is realised by `fast1.py` within 1.4x at depth 1; at
  depth >= 2 and for the RNN it is a model only.
- F falls with the net's accuracy (D1w64 .61: 1.08; D1w96 .79: .87; D1w128 .96: .28): errors of an accurate net sit
  near ties, which a circle integrates exactly.

### T387: fixed (deterministic) great-circle sets on D1w128 (QMC-class; pre-registered at b85cf6a1)

`python score_fixed.py D1w128` (results/fixed_D1w128.json). 12-model D1w128 panel, 1e8-sample truths (seed 8787;
truth SE ~1.8e-5, i.e. F floor ~3e-4), 232 circles per set (C ~4,000 FPE as executed), the same circles for every
model, no random calls.

| design family | F of the 5 fixed sets | mean F | all 5 sets F < 1 |
|---|---|---|---|
| R: random circles (control, `score_fast1.py`) | .280 [.253, .306]; direct .267 | | |
| S: scrambled Sobol, seeds 1-5 | .58, .13, .16, .06, .56 | .30 | yes |
| **H: scrambled Halton, seeds 1-5** | **.03, .28, .19, .10, .14** | **.15** | **yes** |
| P: 23 Sobol points x 10 cyclic shifts | 8.5, 2.2, .25, 1.5, .73 | 2.6 | no |
| K: Korobov lattice, a = 17, 31, 57, 89, 113 | 2.6, .99, .51, 1.7e4, .12 | | no |

Post-hoc sensitivity (not pre-registered; same designs, run after T387): on the D1w128@30 panel (2^30 retrains,
acc .990; random circles F .104) `python score_fixed.py D1w128@30` gives Halton .049, .026, .088, .024, .035 (mean
.045), Sobol .126, .090, .033, .167, .096 (mean .10), cyclic .11-3.5, Korobov 1.8-5.9 (and a = 89 degenerate).

- The per-set bootstrap CIs over models (in the JSON) understate the uncertainty: a fixed set's error is shared by
  the 12 near-identical panel models (|mean error| / RMS error .8-.98 for most sets), so each set is close to one
  draw. The spread across the 5 sets of a family is the honest measure.
- Halton keeps F < 1 on all 5 sets (mean .15, about 2x better than random circles); Sobol on all 5 (mean .30, no
  better than random at n = 232 in 20 dimensions).
- The cyclic-symmetric design fails: 23 base points are too few, and the shifts add no balance the net can use
  (consistent with the cone test: the net is not near-equivariant).
- **Pre-registration defect:** Korobov a = 89 has multiplicative order 4 mod 233, so its 20 lattice coordinates take
  only 4 distinct values; inputs repeat coordinates (exact ties) and the estimate collapses (error .43). The
  generators were fixed without this check (a = 31 and 113 have order 116, a = 17 and 57 order 232).
- A design derived from W1's comparator structure was **not done**: no rule could be stated that does not amount
  to choosing circles by looking at where the net errs.
- Classification: deterministic quadrature over (u, v) in R^20 (QMC-class under project rules, like T10); not
  deductive in ARC's sense.

### Step 2: deductive routes tried (all scoped negatives)

1. **Exact region enumeration with exact masses:** dead at every budget (region table above; 10-D polytope
   Gaussian masses have no closed form on top of that).
2. **Rank-order cones (DRI's idea, `python cone_test.py D1w128_s0_n28 D1w128_s1_n28 D1w128_s2_n28`):** if all
   first-layer units were exact comparators the net would be linear on each of the 10! ordering cones, where the
   label is constant. Measured on the 3 D1w128 seeds: comparator gates disagree with [x_a > x_b] on 2.7-5.3% of
   (unit, input) pairs and on at least one unit for 89-99% of inputs (their small extra weights shift the gate near
   ties); permutation agreement of decisions .93-.95 vs accuracy .95-.97 (no more equivariant than accuracy
   forces); 1.3-1.7k gate subregions per cone (median of 50 cones, 20k samples each; lower bound), so ~5e9 pieces.
   Not tried: an idealised comparator net plus a bounded correction.
3. **Tie-band control variate** (model-free surrogate 1[gap23 > a, gap12 > b] scaled by range): per-sample
   correlation with correctness at most .42 on D3w32 (variance factor <= 1.2); errors are not confined to tie
   bands (63% pick the 3rd, 36% the max; half the error mass lies outside a .01 band). Dropped (exploratory, no
   script kept).
4. Gaussian structure of the first layer: not pursued beyond noting that moment propagation would need ~.2 pp
   accuracy at C = 1e4 FPE, far beyond what such approximations give on this task [inferred].

### Mechanism (short account)

Measured on the 3 D1w128 seeds (logit effects in units of the unit's input norm):
- Every first-layer row sums to ~0 (|sum| / norm <= .015 on every net trained here): the nets learn near-invariance
  to adding a constant to all inputs, which the task has exactly.
- 84-94 of 128 units are pairwise comparators relu(x_a - x_b) (all 45 pairs covered). Each pushes the loser's
  logit down by 27-34 per unit of gap, the winner's down by 5-6, and the other logits up by ~2.
- The other 34-44 units are "both low" detectors relu(-.67 x_a - .57 x_b + .15 sum of the rest), lowering the two
  low inputs' logits by 7-8.
- So logit_i ~ -28 (total amount by which other inputs beat i) - 5 (total amount by which i beats others) + ...:
  an input loses heavily for every input above it and mildly for every input below, so the max is penalised by
  its total lead and the 2nd largest wins unless its gap structure is atypical [inferred from these averages].
- Depth 3, width 32 (seed 0): 24/32 first-layer units are comparators, 8 are "x_a minus a few others".

## Compute accounting
- Training (model construction, not counted in F): 2^28 samples ~6-35 min per run, 2^30 ~25-45 min, one thread.
- Walks are charged in FLOPs of the walk algorithm and converted to FPE of the net being estimated: `walks.py`
  charges 2 FPE (forward-with-slope) per piece plus target/envelope/erf work (erf, atan2, cos, sin = 20 FLOPs);
  `fast1.py` charges its event-driven algorithm (W1 u, W1 v, 2w rank-1 updates, cos/sin per event, 4T per logit
  per arc end, atan2 only at sign changes; numpy's redundant vectorised work is not charged and is marked in the
  code). A conservative count that charges every arc 10 atan2 calls and a midpoint sort gives 44 FPE per walk
  instead of 17.2 (F .72 on D1w128 seed 0 instead of .28).
- Truths: 1e7-sample MC per panel model (seeds 4242 and 4343).

## Claims
- [measured] The simplest bias-free ReLU MLP reaching >= .95 on all 3 seeds (2^28 samples) is one hidden layer of
  128 (2,560 params); widths <= 96 at depth 1, <= 64 at depth 2 and <= 32 at depth 3 fail on at least one seed.
- [measured] At ~432 parameters MLPs reach only .36-.69.
- [measured] Realised region counts of every >= .95 MLP are at least those of M_16,10 (sampled lower bounds).
- [measured] Exact great-circle walks (sampling-class) give F .280 [.253, .306] on the D1w128 panel (direct check
  .267) vs 15.4 [12.5, 18.3] on the M_16,10 panel; deeper MLPs 3.8-5.9.
- [measured] The rank-order cone route fails its two preconditions on D1w128 (comparators not exact; ~1.3k+
  subregions per cone; not near-equivariant).
- [inferred] The gain comes from depth (rank-1 flips with closed-form crossing angles), not from fewer regions.
- [speculative] A deductive estimator for D1w128 is reachable by deriving a deterministic set of great circles
  from the mechanism; untested.

## Alternatives not ruled out
- A deterministic (lattice / designed) set of great circles could keep F < 1; under project rules that is
  sampling-class (QMC), and a mechanism-derived deterministic design is untested.
- 2^28-sample training may undertrain: seeds that failed (D3w28/D3w32 seed 1) might pass at 2^30. The 2^30 retrain
  of D1w128 is running; other configs were not retrained.
- Biased MLPs, other optimisers or longer schedules may reach .95 with fewer parameters.
- The incremental cost model for deeper nets and the RNN is not implemented; a cleverer RNN walk could beat it,
  but the floor (any piece walk, 4T FLOPs per piece) is .29 for M_16,10 vs .16 for D1w128.
- F depends on the panel: copies at relative sigma .001 are close to their seeds; the M_16,10 panel mixes retrains
  of accuracy .88-.95.

## Failure modes
- F of the walk estimators is budget-independent only for C much larger than the per-walk cost (17 FPE for
  `fast1`, ~250-550 FPE for generic circles); at C = 1k FPE the generic walks get only 2-4 walks.
- `walks.py` uses eps-offset evaluation at piece starts; degenerate ties (two breakpoints within 1e-12) could
  misassign a sliver; the Cauchy lines are truncated at |t| = 1e8 (mass 6e-9).
- Region and decision-cell counts use 64-bit hashes; collisions would undercount (negligible at 1e7).
- The ladder used one seed per narrow configuration; a lucky or unlucky seed 0 can misplace the width threshold.
