# T371: rare events on ARC-style weight-noised copies

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) (H-R1) · [Approaches](../README.md)

Owner: Claude child session_014hCDUEjWU4zSYA4jbwe4ty (branch `claude/t371-rare-noise`).

## Question

H-R1 (confirmed, scoped) says the frozen K3 gate-walk estimator gets the far-gap rare error probability
E1(t0) = P(model picks the position of rank 3 AND g23 > t0) within 2x for p ~2e-7 to 2e-4, without sampling. That
was shown on clean retrains only; H-R1 lists "weight-noised copies untested". Does it stay within 2x on copies made
by ARC's own noise protocol (`m16/arc_eval.py`), and where does it break as the noise grows?

## Pre-registration (2026-10-18, pushed before any estimate or truth is computed)

**Models (52).** Bases: M_16,10 itself (`orig`) and three non-triple-tie DEV3 retrains already used for rare events
(`dev3_s526` from T364, `dev3_s528` and `dev3_s530` from T366/T368). For each base: the clean model plus
**4 noise levels sigma = .001, .005, .02, .05** (relative to each matrix's RMS; .05 is arc_eval's default) with
**3 copies each**, drawn by `arc_eval.noisy_models(3, sigma, seed=0)` (orig) or the identical formula on the DEV3
weights (`models.py`; asserted bitwise equal to arc_eval on orig). Same seed at every level, as arc_eval does, so copy
j at the four levels lies on one noise ray. The levels were picked without looking at any rare-event number: .001 and
.005 bracket T337's accuracy-estimator breakdown (.1-.25%), .02 and .05 reach ARC's default.

**Event and thresholds.** E1(t0) at **t0 = 1.0, 1.25, 1.5** (T340's model-free thresholds; truths on the clean
models in H-R1's range). The same thresholds for every model, fixed now.

**Estimator.** The frozen H-R1 estimator, unchanged: K3 = T340c (g12 at 3 exact-law nodes) with the walk fix and the
T354 segment start, ended at the T359 cut **T_hi = 3.5** (declared dropped-mass bound P(g23 > 3.5) = 7.04e-8 per
point, reported, not added). Called exactly as T364's `deep.py`: `t354.est_seg(model, 3, t0s, t2s=(), tmax=3.5)`.
Code read-only from claude/t218-tie-tail-labels @5b4a1250 (t340c.py, t354.py, walkfix.py) and
claude/t286-cheap-frontier @620eb47b (rare.py, cont_g.py); all five sha256 asserted at import (`est_run.py`,
values from r1_cont/t366_freeze_sha256.txt). **Deductive**: no random model inputs. Cost = FLOPs as executed / 5948
(FPE).

**Truth (own code, `truth.py`).** Exact conditional-law MC with a construction different from rare_events/truthc.py
(which draws v2 first): condition on v3, the 3rd largest of 10 iid N(0,1): v3 ~ 360 phi(v) Phi(v)^7
(1 - Phi(v + t0))^2, the two top values iid N | > v3 + t0, the 7 lows iid N | < v3, uniform random positions.
P(g23 > t0) by quadrature (agrees with truthc's v2-first formula to 1e-15 at t0 = 1, 1.25, 1.5, 3.5).
**1e8 conditional samples per t0**, pushed through all 52 models (common random numbers), float64 forward
(checked equal to arc_eval.forward on 20k inputs). Diagnostics per point: share of hits with g23 > 3.5 (the mass the
cut drops) and with g12 < .1 (the triple-tie share that broke K3 in T364/T367). Sanity check: clean orig must agree
with T340's public GPU truths (5.37e-6 / 1.1e-6 / 2.1e-7) within ~3 s.e.

**Order.** (1) push this pre-registration; (2) run all 52 estimates and push them; (3) only then compute truths;
(4) score with `score.py` (written before any truth).

**Go line (fixed now).** Over the 144 noisy (copy, t0) points: **>= 80% within [1/2, 2] = GO**. Points whose truth
has < 30 hits are listed but excluded from the count. Also reported: median ratio, mean log10 ratio (one-signed bias),
counts high/low, within 3x, the subset with truth in H-R1's range [2e-7, 2e-4], and the breakdown by sigma, base and
t0. The 12 clean points are a reference (they repeat T340/T364/T366 conditions with my truth code), not in the go count.

**Expectations and alternatives (rule 3), written before running.**
- (a) Small noise (.001, .005) changes the weights by much less than a retrain does, so the walk (exact for any
  weights) and the model-free closure should carry over: expect within 2x like the clean bases.
- (b) Alternative 1: noise creates errors away from the configurations the R1W closure assumes (lows pinned at
  mu(v4), v4 at one position; T337 saw interior and rank-4+ errors grow with noise; T363/T369 trace E2 and partial
  triple-tie failures to that closure). Then E1 estimates go one-signed LOW as sigma grows, while the truth's
  tie share stays low. Alternative 2: noise moves error mass into the triple-tie sliver (g12 ~ 0), the known K3 failure
  (T364/T367); then misses go low together with a rising tie share in the truth diagnostic. Alternative 3: at large
  sigma the far-gap errors become common (p >> 2e-4) and the event is no longer "rare"; whether K3 holds there is a
  separate question (out of H-R1's range; reported separately). The tie-share and g23 > 3.5 diagnostics are the
  cheap tests that separate 1 from 2 and check the cut.

## Method

- `models.py`: the 52-model panel (above).
- `est_run.py SHARD NSHARD`: frozen estimator on a shard of the panel -> `results/est_SHARD.json`.
- `truth.py T0 N`: conditional-law MC truth -> `results/truth_T0.json` (checkpointed).
- `score.py`: table and verdict -> `results/score.json`.

Environment: `T340_SRC` = checkout of claude/t286-cheap-frontier @620eb47b; `T371_R1C` = r1_cont/ of
claude/t218-tie-tail-labels @5b4a1250; `T371_DEV3` = rare_events/ext/dev3/weights.npz from
claude/t286-cheap-frontier (sha256 62c95203..., asserted).

## Status

Pre-registered; running.

## Results

(pending)

## Compute accounting

(pending) Estimator: FLOPs as executed / 5948. Truth: 3 x 1e8 conditional samples x 52 models (sampling, truth
only; not part of the estimator's budget).

## Claims

(pending)

## Alternatives not ruled out

(pending)

## Failure modes

(pending)
