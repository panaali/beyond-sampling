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

## Method
`s1models.py` (S1Block subclasses the harness Block; S1Transformer subclasses TinyTransformer, per-layer types),
`s1train.py` (harness recipe + schedules + modsum curriculum), `load_s1.py` (loader). A model whose layers are all
`hard` with tol 1e-6 is bit-identical in forward to the harness `hard` (checked: max |diff| 0 on random inputs), so
those checkpoints load in the harness too.

## Status
Pre-registered; training not started.
