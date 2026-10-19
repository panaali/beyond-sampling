# T444 / mini-LM S1: exactness-preserving attention trained to softmax accuracy

Spec: [STREAMS.md S1](../mini_lm/STREAMS.md) (branch `claude/t433-mini-lm` d45e9d21). Harness used read-only through
`sys.path` (`models.py`, `langs.py`, `metrics.py`); nothing in `../mini_lm/` is edited.

## Question
Can an attention whose routing is piecewise constant or piecewise linear in the scores (so our exact tools still
apply) match softmax on dyck2 at L = 32, and can an exactness-friendly variant learn modsum at n = 20?

## Pre-registration (committed before any training run)

**Reference (softmax baselines, T433, 1e6 samples at L 32):** acc .99998 / .99999 / .99999; B0.5 3e-6 / 6e-6 / 1e-6
(mean 3.3e-6).
**dyck2 pass line, per seed:** acc >= .98999 (mean softmax acc - 1 pp) AND B0.5 <= 3.3e-5 (10x the mean softmax
B0.5), both on 1e6 L-32 samples (`metrics.sampled`, seed 7e6 + seed). **Success = 3/3 confirmation seeds pass**
for a variant whose routing is piecewise constant (hard / AHAT / key-only hard / pos+hard) or piecewise linear
(sparsemax / key-only sparsemax / pos).
**modsum pass line:** acc >= .95 at n = 20 (1e6 samples) on 2/3 confirmation seeds, for an exactness-friendly
variant (pos, hard, sparsemax, or mixtures of these).

**Seeds.** Development: seed 0 only, screened on 2e5 samples. Confirmation: fresh seeds 10, 11, 12, 1e6 samples
plus exact L = 15 (dyck2) / n <= 10 (modsum) enumeration. Only confirmation seeds count for the verdict.

**Recipe common to all:** harness recipe (d 32, mlp 64, no LN, AdamW lr 3e-3 cosine, wd .01, batch 256, clip 1,
2 threads), 2 layers unless stated, 8000 steps unless stated. Eval of hard layers is pure hard (tol as stated).

**dyck2 development round 1 (seed 0):**
| id | attn (layer1,layer2) | training |
|---|---|---|
| D1 | hard,hard (tol 1e-6) | anneal: softmax(beta s), beta 1 -> 30 geometric over 75%, last 25% straight-through |
| D2 | hard,hard (tol 1e-6) | mix: (1-lam) softmax(30 s) + lam ST, lam 0 -> 1 over 75%, then ST |
| D3 | hard,hard (AHAT tol .5) | anneal as D1 |
| D4 | sparsemax,sparsemax | plain (sparsemax train and eval) |
| D5 | pos,hard | anneal as D1 (layer 2) |
| D6 | pos,sparsemax | plain |
| D7 | keyhard,keyhard | anneal as D1 |
| D8 | keysparse,keysparse | plain |
| D9 | pos (1 layer) | plain |
| D10 | pos,pos with mlp 64, 16000 steps | plain (more steps for the .996 baseline) |

**Rounds 2-3 (only if no dev variant passes the pass line at 2e5 samples; at most 40 trainings in all):** for the
top 2 dev variants by (acc, then lower B0.5): (a) 16000 steps; (b) beta1 100 with hard_frac .4; (c) lr 1e-3;
(d) 3 layers. The best variant(s) meeting the dev pass line (at most 2) go to confirmation, chosen by the rule above
before confirmation seeds are trained.

**modsum (seed 0 dev, curriculum n_max 2 -> 20 linear over the first 50% of 16000 steps unless stated):**
M1 pos,pos; M2 pos,hard (anneal); M3 hard,hard AHAT tol .5 (anneal); M4 sparsemax,sparsemax; M5 softmax,softmax
(reference only, not a candidate); M6 pos,pos without curriculum, 16000 steps (to separate curriculum from steps).
Confirmation: the best exactness-friendly variant with dev acc >= .95 on seeds 10-12.

**Predictions [speculative]:** P-S1a annealing closes most of the hard gap (D1 acc >= .99) but B0.5 stays above
3.3e-5; P-S1b pos,hard is the best piecewise-constant variant (layer 1 averaging computes depth, layer 2 matches
it); P-S1c key-only fails (< .95: matching needs the query's depth); P-S1d the curriculum makes pos reach >= .95 on
modsum.

## Round 1 results (dev, seed 0, 2e5 samples; `logs/r1.log`) and the selection that follows from the rule

| id | attn | acc L32 | illegal mass | B0.5 | dev pass? |
|---|---|---|---|---|---|
| D1 | hard,hard anneal | .9165 | 6.5e-2 | 0 | no (acc) |
| D2 | hard,hard mix | .9288 | 4.7e-2 | 7.6e-3 | no |
| D3 | hard AHAT tol .5 anneal | .9356 | 4.3e-2 | 5.2e-3 | no |
| D4 | sparsemax,sparsemax | .9890 | 6.9e-3 | 7.1e-4 | no |
| D5 | pos,hard anneal | .99991 | 6.6e-5 | 2.9e-4 | no (B0.5) |
| D6 | pos,sparsemax | .99999 | 1.6e-5 | 1.0e-5 | **yes** |
| D7 | keyhard,keyhard anneal | .8544 | 8.7e-2 | 1.1e-4 | no |
| D8 | keysparse,keysparse | .99905 | 1.3e-3 | 5.9e-4 | no (B0.5) |
| D9 | pos (1 layer) | .9528 | 3.0e-2 | 1.4e-3 | no |
| D10 | pos,pos 16k steps | .99804 | 1.7e-3 | 7.5e-4 | no (B0.5) |

modsum n = 20: M1 pos,pos + curriculum **.9985**; M2 pos,hard + curriculum .9905; M3 hard AHAT .1996 (chance);
M4 sparsemax .1994 (chance); M5 softmax + curriculum .9238; M6 pos,pos without curriculum .2124 (chance).

**Selection by the pre-registered rule:** D6 goes to dyck2 confirmation (seeds 10-12) and M1 to modsum confirmation.

**Addendum A (written after round 1, before any further run; exploratory, does not change the verdict rule):**
since no piecewise-*constant* variant passed, and D5 (pos,hard) misses only on B0.5, run pre-registered round-2
options on D5 as a labelled extra: R2a pos,hard anneal 16000 steps; R2b pos,hard anneal beta1 100 hard_frac .4;
R2c pos,pos,hard anneal (3 layers). Any R2 variant that meets the dev pass line also gets confirmation seeds 10-12,
reported as a second (post-hoc chosen) candidate. Running total after this: 16 + 6 confirmation + 3 + <= 3 = <= 28
trainings (cap 40).

## Round 2 results (`logs/r2.log`)

Confirmation (fresh seeds, 1e6 samples at the large horizon; exact = full enumeration at L 15 / n <= 10):
| model | acc large | illegal mass | B0.5 large | B0.9 large | exact acc | exact B0.5 | pass? |
|---|---|---|---|---|---|---|---|
| dyck2 D6 pos,sparsemax s10 | .99999 | 1.6e-5 | 1.0e-5 | 0 | 1.000000 | 0 | yes |
| dyck2 D6 s11 | .99999 | 1.8e-5 | 2.0e-6 | 0 | 1.000000 | 0 | yes |
| dyck2 D6 s12 | .99999 | 1.6e-5 | 5.0e-6 | 0 | 1.000000 | 0 | yes |
| modsum M1 pos,pos curric s10 | .9633 | 7.5e-2 | 6.0e-2 | 1.4e-3 | .99822 | 3.0e-3 | yes |
| modsum M1 s11 | .99946 | 5.6e-3 | 5.8e-4 | 2.3e-4 | .99995 | 6.4e-5 | yes |
| modsum M1 s12 | .9776 | 5.3e-2 | 3.2e-2 | 3.3e-3 | .99997 | 2.7e-5 | yes |

Addendum A (dev, seed 0, 2e5): R2a pos,hard 16k steps acc .99992, B0.5 8.0e-5; R2b beta1 100 acc .99916, B0.5
9.3e-4; R2c pos,pos,hard acc .99998, B0.5 4.5e-5 (9 events; just above 3.3e-5). None passes the dev line.

**Addendum B (written after round 2, before running):** combine the two helpful changes: R3a pos,pos,hard anneal
16000 steps; R3b the same with hard_frac .4. A variant that meets the dev line goes to confirmation on seeds 10-12
(reported as a post-hoc piecewise-constant candidate). Cap: 25 + 2 + <= 3 = <= 30 trainings. If none passes, the
piecewise-constant result is a scoped negative.

## Round 3-4 results (`logs/r3.log`, `logs/r4.log`)

Dev (seed 0, 2e5): R3a pos,pos,hard 16k steps acc .99999, B0.5 1.0e-5 (pass); R3b hard_frac .4 acc .99998, B0.5
1.5e-5 (pass). R3a (better) to confirmation:
| model | acc L32 | illegal mass | B0.5 L32 (1e6) | exact L15 acc | exact L15 B0.5 | pass? |
|---|---|---|---|---|---|---|
| dyck2 R3a pos,pos,hard s10 | .99997 | 2.8e-5 | **3.5e-5** | 1.000000 | 0 | **no** (B0.5 > 3.3e-5, 35 vs 33 events) |
| dyck2 R3a s11 | .99999 | 9.1e-6 | 2.1e-5 | 1.000000 | 0 | yes |
| dyck2 R3a s12 | .99999 | 1.0e-5 | 1.5e-5 | 1.000000 | 0 | yes |

## Method
`s1models.py` (S1Block subclasses the harness Block; S1Transformer subclasses TinyTransformer, per-layer types),
`s1train.py` (harness recipe + schedules + modsum curriculum), `load_s1.py` (loader), `run_jobs.sh` + `jobs_r*.txt`
(every run, 2 processes under `ulimit -v 2500000`). A model whose layers are all `hard` with tol 1e-6 is
bit-identical in forward to the harness `hard` (max |diff| 0 on random inputs), so those checkpoints load in the
harness too. Scoring is the harness `metrics.sampled` / `metrics.exact_small` unchanged. Sampling-based evaluation
(plain MC over the language law) plus exact enumeration at the small horizon; no estimator is scored here.

**Frozen recipes (deliverables, checkpoints in `models/`):**
| name | attn | params | FLOPs/FPE (large) | recipe |
|---|---|---|---|---|
| `dyck2_D6_s{10,11,12}` | pos, sparsemax | 16,932 | 1.03e6 | 8000 steps, plain |
| `modsum_M1_s{10,11,12}` | pos, pos | 14,638 | 5.8e5 | 16000 steps, curriculum n_max 2 -> 20 over first 50% |
| `dyck2_R3a_s{10,11,12}` (near miss, 2/3) | pos, pos, hard | 24,228 | 1.45e6 | 16000 steps, anneal beta 1 -> 30 over 75%, then ST |

```python
import sys; sys.path.insert(0, "m16/approaches/mini_lm_s1")
from load_s1 import load
model, lang = load("dyck2_D6_s10")   # eval mode; works with ../mini_lm/metrics.py
```

**Which exact tool each variant keeps** [inferred from the architecture]:
| variant | routing | the network given the inputs |
|---|---|---|
| pos (any layer) | input-independent constant mixing | linear in the embeddings |
| pos,pos (M1) | constant | piecewise linear in the one-hot inputs (only ReLU MLPs): the strongest case |
| pos,hard (R3a, D5) | piecewise constant (argmax set of bilinear scores) | piecewise linear given the routing; branch on routing (finite) |
| pos,sparsemax (D6) | piecewise linear in the scores (weights = s - tau on the support set) | given (support set, ReLU pattern): polynomial (degree 3) in the layer-2 residual, which is piecewise linear in the inputs; branch on the support set |
| sparsemax / keysparse | as D6 | as D6 |

## Status
**Done.** Pre-registered dyck2 success met by a piecewise-*linear*-routing variant (D6 pos,sparsemax, 3/3); modsum
success met (M1 pos,pos + curriculum, 3/3). The piecewise-*constant* variant (R3a pos,pos,hard) is a near miss
(2/3; seed 10 B0.5 3.5e-5 vs line 3.3e-5): scoped negative (unverified).

## Results
Tables above. Reproduce any row: `python s1train.py <args from jobs_r*.txt>` (e.g.
`python s1train.py dyck2 D6 pos,sparsemax 10 --eval_n 1000000 --exact`); logs in `logs/`, full metrics in
`models/<name>.json`. Softmax reference: T433 `results_baselines.md`.

## Compute accounting
30 trainings (16 round 1, 9 round 2, 2 round 3, 3 round 4; cap 40). Training + scoring wall time summed over
processes: 3.0 h (`models/*.json` train_seconds + eval_seconds), each process 2 threads, so <= 6 CPU-h. Model
evaluations: 1.32e7 sampled large-horizon sequences (21 x 2e5 + 9 x 1e6), plus 6 dyck2 exact enumerations at
L 15 (7.4e6 sequences each) and 3 modsum exact (1.2e7 each). No GPU.

## Claims
- [measured] pos,sparsemax (D6) matches softmax on dyck2 at L 32 on 3/3 fresh seeds: acc .99999 (softmax
  .99998-.99999), B0.5 1e-5 / 2e-6 / 5e-6 (softmax 1e-6 - 6e-6), exact L 15 acc 1 and B0.5 0.
- [measured] A curriculum on n makes the position-only transformer (pos,pos) learn modsum at n 20: acc .963 / .9995 /
  .978 on seeds 10-12; without it the same model stays at chance (M6 .21, seed 0). Softmax with the same curriculum
  reached .924 (seed 0).
- [measured] Hard attention: annealing / mixing alone does not fix the 2-layer hard model (D1-D3 .917-.936); putting
  a pos layer first does (pos,hard .9999), and pos,pos,hard reaches softmax accuracy (.99997-.99999) but its B0.5
  (1.5e-5 - 3.5e-5) is 5-10x softmax's; 2/3 seeds pass the 10x line.
- [inferred] The hard models' bottleneck was layer 1: unique-argmax attention cannot average (depth counting needs a
  uniform average), and a pos layer can. Supported by D9 (pos 1 layer .953) and D5 vs D1, not tested directly.
- [measured] Key-only scores: keyhard fails (.854); keysparse reaches .999 but B0.5 5.9e-4.
- [speculative] More steps or a 4th layer would bring pos,pos,hard to 3/3.

## Alternatives not ruled out
- D6's and R3a's B0.5 at 1e6 samples have relative SE ~30-45%; R3a s10's miss (35 vs 33 events) is within noise
  of the line; a 1e7-sample re-score could flip it either way (the verdict stays as pre-registered).
- Seed 0 was used for dev selection; the confirmation seeds are fresh, but the variant set was extended twice
  (addenda A, B) after seeing results; R3a is a post-hoc candidate.
- modsum dev used one seed per variant: M3/M4 (hard AHAT, sparsemax) at chance may be seed or lr, not the
  architecture.
- No check of length generalisation (L > 32) or of n > 20.

## Failure modes
- pos layers have a fixed Lmax (32 / 22): models cannot run longer sequences.
- sparsemax ties / support changes make the piecewise structure finer than hard routing; exact branching over
  support sets may be costlier than over argmaxes.
- modsum M1 s10 and s12 have large-horizon B0.5 3-6% (illegal answer mass rises at large n) though accuracy passes.

## Exhaustion checklist (piecewise-constant negative)
Tried: straight-through (T433), anneal (D1), mix (D2), AHAT tol .5 (D3), pos,hard (D5, R2a 16k, R2b beta1 100),
key-only hard (D7), pos,pos,hard (R2c, R3a, R3b). Not tried: 4 layers; > 16k steps; lr sweep; AHAT tolerance
with a pos first layer; multi-head (pos head + hard head in one layer); distilling softmax into hard; a 1e7-sample
re-score of R3a s10. Most promising: pos,pos,hard at 32k steps or with distillation from D6/softmax.
