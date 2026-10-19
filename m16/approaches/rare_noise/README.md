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

**Done (2026-10-19); follow-up T388 (graded g12 on the misses) done below: GO, 16/16 misses recovered.** Estimates sealed and pushed first (@541010cf), then truths and scoring. Verdict by the
pre-registered line: **GO, 128 / 144 noisy points within 2x (88.9%; line 80%)**. Unverified: nobody has re-checked it.

## Results

Commands: `python est_run.py 0 2` and `python est_run.py 1 2` (estimates); `python truth.py 1.0 1e8`,
`python truth.py 1.25 1e8`, `python truth.py 1.5 1e8` (truths); `python score.py` (`results/score.txt`,
`results/score.json`); `python analysis.py` (post hoc, `results/analysis.json`). Environment as under Method.

**Pre-registered summary (all 144 noisy points; none excluded, every truth has >= 1,000 hits; truth rel. s.e. <= 2%).**

| set | n | within 2x | within 3x | median est/truth | mean log10 ratio | high / low |
|---|---|---|---|---|---|---|
| **all noisy** | 144 | **128 (88.9%)** | 133 | **1.02** | -.09 | 81 / 63 |
| sigma .001 | 36 | 28 (78%) | 29 | 1.01 | -.26 | 18 / 18 |
| sigma .005 | 36 | 28 (78%) | 32 | 1.02 | -.13 | 19 / 17 |
| sigma .02 | 36 | **36 (100%)** | 36 | 1.03 | .00 | 20 / 16 |
| sigma .05 | 36 | **36 (100%)** | 36 | 1.02 | .01 | 24 / 12 |
| truth in H-R1's range [2e-7, 2e-4] | 77 | 61 (79%) | 66 | .98 | -.19 | 35 / 42 |
| t0 = 1.0 / 1.25 / 1.5 | 48 each | 42 / 43 / 43 | 43 / 45 / 45 | 1.02 / 1.03 / 1.01 | | |
| clean bases (reference) | 12 | 8 | 9 | 1.01 | -.24 | 6 / 6 |

**Per base and sigma** (est/truth, ordered copy 0 t0 1.0 / 1.25 / 1.5, then copy 1, then copy 2; "tie" = share of the
truth's hits with g12 < .1, min-max over the row):

| base | sigma | truth range | est/truth | tie % |
|---|---|---|---|---|
| orig | 0 | 2.2e-07 - 5.4e-06 | 1.75 1.25 0.47 | 81-92 |
| orig | .001 | 1.4e-07 - 5.7e-06 | 1.55 1.05 0.38 · 1.61 1.50 0.62 · 1.79 1.24 0.52 | 80-94 |
| orig | .005 | 5.9e-07 - 3.8e-05 | 0.39 0.47 0.79 · 1.01 1.05 0.67 · 1.24 0.68 0.46 | 13-88 |
| orig | .02 | 5.0e-05 - 1.4e-03 | 0.94 0.84 0.84 · 1.04 1.04 1.00 · 0.90 0.87 0.77 | 10-18 |
| orig | .05 | 2.1e-04 - 2.8e-03 | 1.05 1.03 1.02 · 1.02 1.02 1.03 · 0.97 0.97 0.97 | 18-21 |
| dev3_s526 | 0 | 1.4e-05 - 8.7e-05 | 0.79 1.01 1.08 | 67-77 |
| dev3_s526 | .001 | 8.6e-06 - 7.1e-05 | 1.11 0.99 0.90 · 1.00 0.94 1.08 · 0.78 1.05 1.22 | 48-74 |
| dev3_s526 | .005 | 1.6e-05 - 3.9e-04 | 1.09 1.14 1.09 · 1.26 1.04 0.93 · 0.74 0.83 0.81 | 15-34 |
| dev3_s526 | .02 | 7.1e-05 - 4.2e-03 | 1.10 1.11 1.09 · 1.13 1.10 1.09 · 0.88 0.90 0.94 | 14-22 |
| dev3_s526 | .05 | 4.5e-04 - 4.7e-03 | 1.04 1.02 1.01 · 1.18 1.17 1.15 · 0.97 0.96 0.96 | 19-22 |
| dev3_s528 | 0 | 2.6e-06 - 1.5e-05 | **0.08 0.08 0.20** | 94-96 |
| dev3_s528 | .001 | 1.1e-06 - 2.6e-05 | **0.26** 0.52 1.29 · **0.08 0.10 0.23** · **0.05 0.01 0.0015** | 78-100 |
| dev3_s528 | .005 | 2.5e-06 - 1.3e-04 | 1.31 1.52 1.35 · **0.21 0.37** 0.92 · **0.15 0.12 0.12** | 25-93 |
| dev3_s528 | .02 | 3.2e-06 - 1.2e-03 | 1.05 1.05 1.02 · 0.61 1.16 1.43 · 0.95 0.92 0.94 | 12-41 |
| dev3_s528 | .05 | 4.2e-05 - 3.1e-03 | 1.02 1.01 1.00 · 0.89 0.94 0.86 · 1.08 1.09 1.07 | 8-22 |
| dev3_s530 | 0 | 4.6e-05 - 3.6e-04 | 1.00 1.02 1.04 | 31-33 |
| dev3_s530 | .001 | 4.3e-05 - 4.2e-04 | 1.03 1.03 1.04 · 1.09 1.18 1.21 · 0.95 0.99 1.05 | 31-36 |
| dev3_s530 | .005 | 4.2e-05 - 7.6e-04 | 1.17 1.14 1.13 · 1.07 1.11 1.09 · 1.02 0.98 1.03 | 26-33 |
| dev3_s530 | .02 | 1.9e-04 - 2.2e-03 | 1.20 1.20 1.17 · 1.08 1.09 1.09 · 0.97 0.97 0.99 | 16-24 |
| dev3_s530 | .05 | 4.3e-04 - 5.1e-03 | 1.18 1.16 1.12 · 1.06 1.09 1.09 · 0.96 0.98 0.98 | 18-23 |

**Where it breaks.** All 16 noisy misses are LOW, and all lie at small noise (.001, .005): 12 on dev3_s528's copies
(down to .0015x), 4 on orig's copies (.38-.47). None at .02 or .05. Larger noise does not break the estimator; it
removes the conditions it fails in.

**Post hoc (not pre-registered; `analysis.py`): the misses track the truth's triple-tie share.** Over all 156 points
(noisy + clean):

| truth share of hits with g12 < .1 | n | within 2x | median ratio |
|---|---|---|---|
| < 50% | 110 | **110** | 1.03 |
| 50-80% | 15 | 13 | 1.00 |
| >= 80% | 31 | 13 | .38 |

Noise at .02-.05 spreads the error mass out of the g12 ~ 0 sliver (tie share 8-41%), and the error rate rises 10-300x.
In that regime K3 is within 1.2x almost everywhere. A high tie share is necessary for a miss here but not sufficient:
clean orig at t0 1.0 / 1.25 (81-89%) is still within 2x.

**Checks.**
- Truth code vs T340's public GPU truths on clean orig (1e10 samples): 5.44e-6 vs 5.37e-6, 1.07e-6 vs ~1.1e-6,
  2.16e-7 vs ~2.1e-7 (within ~1 s.e.). My clean-orig ratios (1.75 / 1.25 / .47) reproduce T340c's K3 (1.73-1.77 /
  1.24 / .48-.49).
- The cut: the truths' share of hits with g23 > 3.5 is at most 1.2e-5 on any point. So the T_hi = 3.5 cut drops nothing
  that matters. The declared bound 7.04e-8 is >= 1/3 of the truth only for orig-family points at t0 = 1.5.
- Estimator hashes: all five asserted at import. The noise replica is bitwise equal to arc_eval on orig.

**New fact about a model already in use.** Clean dev3_s528, used as "non-triple-tie" in T366/T368 at deep thresholds
(2.35 / 2.75, where K3 was 1.26-1.39), IS triple-tie-dominated at t0 = 1.0-1.5: 94-96% of its E1 hits have
g12 < .1, and K3 is .08 / .08 / .20 there [measured]. So "triple-tie-dominated" depends on the threshold, not only on
the model.

## Compute accounting

- **Estimator (deductive, no random inputs):** FLOPs as executed / 5948. Per model (all 3 t0 in one call) 0.31-0.55M FPE,
  median 0.37M; 19.5M FPE for all 52 models. Wall time 30-65 s per model on one core. Cost rises mildly with
  sigma (more gate cells on the lines).
- **Truth (sampling, truth only, not part of any budget):** 3 x 1e8 conditional samples, each pushed through all 52
  models (1.56e10 model forwards), float64, about 1 core-hour per t0.
- Equal compute (context, not scored): at ~0.37M FPE, plain MC expects p x C events, i.e. < 1 event for every point
  with p < 2.7e-6 (orig and dev3_s528 at small sigma), and 2-1,900 events elsewhere. At sigma .02-.05 (p 5e-5 to 5e-3),
  sampling at the same cost would beat K3's ~10% error. This test did not score that.

## Claims

1. [measured] The frozen H-R1 estimator (K3, T_hi 3.5, unchanged) is within 2x of an independent truth on 128 / 144
   (88.9%) arc_eval noisy copies of 4 bases at E1(t0 = 1, 1.25, 1.5), median est/truth 1.02, at 0.31-0.55M FPE per model
   (3 thresholds). The pre-registered line (80%) is passed.
2. [measured] At sigma .02 and .05 (ARC's default), 72 / 72 points are within 2x (range .61-1.43;
   69 / 72 within 1.25x). The misses (16) are all at sigma <= .005 and all low.
3. [measured, post hoc] Every point whose truth has < 50% of its hits at g12 < .1 is within 2x (110 / 110). Points at
   >= 80% are within 2x on only 13 / 31 (median .38). [inferred] The failures are the known triple-tie sliver failure
   of 3 g12 nodes (T364/T367), not an effect of noise. Noise affects the result only through how much error mass it
   leaves in the sliver.
4. [measured] dev3_s528 is triple-tie-dominated at t0 = 1.0-1.5 (clean: 94-96%, K3 .08-.20), unlike at T366's deep
   thresholds.
5. [inferred] H-R1's scope item "weight-noised copies untested" can be narrowed: on non-triple-tie points, noisy copies
   behave like clean retrains. The triple-tie limitation H-R1 already records still applies, and small noise does not
   remove it.
6. [measured] Within H-R1's p range [2e-7, 2e-4], 61 / 77 = 79% are within 2x, just under 80% (secondary, not the go
   line). The range is where the small-noise, sliver-dominated points live.

## Alternatives not ruled out

- **Common noise rays.** arc_eval uses the same seed at every sigma, so copy j at .001 / .005 / .02 / .05 shares one
  direction. The 4 levels are not independent samples of noise directions; the per-sigma breakdown is 3 directions
  per base. Per-copy consistency (e.g. dev3_s528 copy 0 good at every sigma, copies 1-2 bad at small sigma) is partly
  one direction seen four times.
- **Islands sliding across fixed lines (DRI FYI from T372, claude/t372-rare-gradient 1973bfb9, k3_moved.json; not read
  by me).** K3 can be right at the trained weights but 1.3-1.75x off at nearby weights, because the islands move
  relative to its fixed walk lines. That fits orig@.001 (1.05-1.79 at t0 1.0 / 1.25, vs 1.25-1.75 clean) and some of
  the 1.1-1.5x scatter at small sigma. It may also contribute to the low misses, which this test cannot separate from
  the triple-tie mechanism. The tie-share split (claim 3) is post hoc and correlational.
- **Selection of bases.** dev3_s526/s528/s530 were picked as "non-triple-tie" from T366/T368 at deep t0. dev3_s528 turned
  out not to be at these t0 (claim 4). With another base set, the small-sigma fraction could be higher or lower. 4 bases
  is a small panel.
- **Large-sigma success may be partly "easy" events.** At sigma .02-.05 the event is far less rare (p up to 5e-3) and the
  tie share is low, so this is a different regime from H-R1's. Passing there does not show the estimator is good at
  rare noisy events with a large sliver share.
- **Truth correlation.** All 52 models share the same draws per t0. Each truth is unbiased, but errors are correlated
  across models; with rel. s.e. <= 2% this cannot move any ratio across the 2x line.
- Not tried: the graded-g12 rule (T366 `estimator_t366`) on the missed points (the obvious next check: does it fix
  dev3_s528 at small sigma as it fixed s541?); E2; more copies per level or independent seeds per level; sigma between
  .005 and .02 (where the transition happens).

## Failure modes

- Triple-tie-dominated far-gap errors (truth g12 < .1 share >~ 80%): K3 misses low, by up to 650x (dev3_s528@.001#2,
  t0 1.5: .0015). Small noise (sigma <= .005) keeps or creates such configurations; at sigma >= .02 they wash out on all
  4 bases here.
- The model-free t0 does not control which regime a model is in: the same model can be sliver-dominated at one t0 and not
  at another (dev3_s528).
- The T_hi = 3.5 bound (7.04e-8) is not small next to truths of ~1-2.5e-7 (orig family at t0 1.5). Measured dropped
  mass is negligible, but the rigorous bound is loose there.

---

# T388: the graded-g12 rule (T366) on T371's misses

## T388 pre-registration (2026-10-19, pushed before any T388 estimate is computed)

**Question.** T371's 16 noisy misses are all low, all at sigma <= .005, and all on points where >= 65% of the truth's
hits have g12 < .1. Does T366's graded g12 rule (10 Gauss-Legendre nodes clustered on g12 in [0, .4] plus a 3-node law
rule beyond), the known fix for the triple-tie sliver, recover them without breaking the points K3 already gets?

**Estimator.** `estimator_t366.est(model, (1.0, 1.25, 1.5))` from claude/t218-tie-tail-labels @5b4a1250, run
unmodified (its frozen settings: fixed walk, T354 segment start, T_hi = 6). All 7 sha256 in `t366_freeze_sha256.txt`
asserted at import (`est_t388.py`). Deductive. Run on all 52 models of T371's panel (one call gives the 3 t0).

**Truths.** T371's, unchanged (`results/truth_*.json`; already known to me, so this is a check of a frozen
estimator on known truths, not a blind test; no setting is chosen by me).

**Sets (fixed by T371's scoring, `results/score.json`).** Misses: the 16 noisy (copy, t0) points with K3 outside
[1/2, 2]. Control: the 128 noisy points with K3 inside.

**Go (DRI, fixed now).** >= 12 / 16 misses within 2x AND <= 6 / 128 former hits outside 2x. Reported by the truth's tie
share (< 50%, 50-80%, >= 80%), because T367/T369 found the graded rule over-estimates where triple-tie hits are a
large but partial share (s547, 80%: 2.43 / 2.12; traced to the R1W closure). The 12 clean points and the cost
(FPE, vs K3's) are reported. Scoring: `score_t388.py` (written before any estimate).

**Expectations / alternatives.** (a) If the misses are g12 quadrature (T364's diagnosis), the rule lifts them toward 1.
(b) If they are partly the closure (lows pinned at mu(v4), fixed v4 position; T369), orig-family misses (tie 65-92%)
may stay low or overshoot as s547 did. (c) The control may go high where tie shares are 50-80% (T367's s547 pattern).

## T388 deviation from the pre-registration (disclosed before scoring)

The frozen code hit the 2 GB per-process cap (`ulimit -v 2500000`; numpy MemoryError in walkfix.py) at model 20 of 52:
it walks all of a region's lines at once, and the noisy copies have more gate cells. Peak RSS was ~1.8 GB in the first
seconds of a fresh process. The remaining 32 models were run through `t388_chunked.py`. It replicates `t366.est_e1`'s
loop but walks each region's lines in chunks of 40,000 rows, calling the frozen `walk_seg` and `rare.mass` unmodified.
Rows are independent, so only the summation order changes.
- **Check** (`python est_t388.py --check dev3_s526@0.05#2,dev3_s526@0.001#0`, `results/t388_chunk_check.json`): estimates
  equal the unchunked frozen run to <= 2.2e-16 relative; C is +522 FPE (+0.02%, per-chunk fixed charges); peak RSS 0.8 GB.
- Which models used which path: field `chunked` in `results/t388_est_*.json` (20 unchunked, 32 chunked).

## T388 results: GO (unverified; `python score_t388.py` -> `results/t388_score.txt`, `results/t388_score.json`)

| set | n | K3 within 2x | **graded within 2x** | K3 median | graded median |
|---|---|---|---|---|---|
| T371's misses (go: >= 12) | 16 | 0 | **16** | .18 | .89 |
| T371's hits (go: <= 6 broken) | 128 | 128 | **127 (1 broken)** | 1.03 | 1.05 |
| all noisy | 144 | 128 | **143** | 1.02 | 1.04 |
| truth tie share < 50% | 107 | 107 | 107 | 1.03 | 1.05 |
| tie share 50-80% | 12 | 10 | 11 | .97 | 1.08 |
| tie share >= 80% | 25 | 11 | **25** | .38 | .96 |
| clean bases (reference) | 12 | 8 | **12** | 1.01 | 1.07 |

- **Misses, K3 -> graded:** dev3_s528 copies .002-.37 -> .82-1.06 (12/12, e.g. @.001#2 t0 1.5 .0015 -> .99). orig
  copies .38-.47 -> .52-.88 (4/4; 2 of them only just inside, .52 and .57).
- **The one broken hit:** orig@.005#0, t0 1.5 (tie share 57%): .79 -> .36.
- **No overshoot anywhere** (max graded ratio 1.72, orig@.001#1 t0 1.0). T367's s547 pattern (2.1-2.4x high at a large
  but partial tie share) does not appear on this panel.
- Clean dev3_s528 .08 / .08 / .20 -> .82 / 1.02 / 1.08; clean orig t0 1.5 .47 -> 1.05.
- **Cost:** 2.17-3.96M FPE per model (3 thresholds), median 7.1x K3's here. Part of that is the frozen settings: the
  graded estimator walks to T_hi = 6, while T371's K3 stops at 3.5.

**T388 claims.**
1. [measured] The frozen graded-g12 estimator recovers all 16 of K3's noisy-copy misses within 2x. It moves 1/128 of
   K3's hits outside 2x. The pre-registered go line (>= 12/16 and <= 6/128) is met.
2. [measured] Over all 144 noisy points it is within 2x on 143 (median 1.04), and on 25/25 sliver-dominated
   (>= 80%) points.
3. [inferred] T371's misses were the g12 quadrature at the triple-tie sliver, as diagnosed. The orig-family residual
   (.36-.88, mostly low, at partial tie shares) fits T369's closure limit (lows pinned at mu(v4)) but was not tested here.
4. [measured] The cost is 7x K3's at these settings. At 2-4M FPE, plain MC expects p x C >= 1 event on most points
   (p >~ 5e-7), so this is an accuracy result, not an equal-compute win.

**T388 alternatives / limits.** The truths were known to me (T371), but the estimator is frozen code and no setting
was chosen, so this checks the code, not the selection. The panel is the same 4 bases and the same noise rays as
T371. The graded rule's known overshoot regime (partial tie share ~80% with a closure artifact, s547) is not
represented here, so this does not calibrate the rule there. Not tried: T_hi = 3.5 for the graded rule (a cost
lever), E2.

---

# T404: the graded rule's cost lever, and the interior event E2, on the noisy copies

## T404 pre-registration (2026-10-19, pushed before any T404 estimate or E2 truth)

Same 52-model panel as T371/T388 (not T403's models or folder). Everything below is deductive on the estimator side;
frozen code is called read-only with its hashes asserted (`t404_est.py` imports `est_t388.py`'s 7-hash check).

**F (both parts).** Per point, est rel. error r = |est/truth - 1|; sampling at the estimator's own executed cost C (FPE)
has relative error s = sqrt((1 - p) / (p C)), capped at 1 (the zero answer) when p C < 1 (T340's convention).
F = r^2 / s^2. Reported: counts with **F <= 1 (strict)** and **F <= 10 (lenient)**, and the median F. F is secondary;
est/truth within 2x is primary.

### Part (a): graded g12 at T_hi = 3.5 (cost lever)

- Estimator: `t366.est_e1`'s loop (via the T388 chunked wrapper, which matched the frozen run to 2e-16) with
  `tmax = 3.5` instead of the frozen 6. That is the one change; it is T359's cut applied to the graded rule. Declared
  dropped-mass bound per point P(g23 > 3.5) = 7.04e-8 (reported, not added).
- Truths: T371's (known to me; this is a code check, not a blind test).
- **Go: >= 140 / 144 noisy points within 2x.** Also reported: C / K3's C (T371) and C / T388's C, per model (median,
  range); the change in est vs T388 (expected <= the bound).

### Part (b): the interior event E2 on the noisy copies

- **Event:** E2(t) = {network wrong (any pick), g12 >= t, g23 >= t}, at **t = 0.5, 0.75, 1.0** (T340's .5 / .75 plus
  1.0 as in T360; fixed now, model-free).
- **Estimators (both frozen, both reported):**
  - **K3-E2** (primary): the E2 branch of `t354.est_seg` (T340c lines at 3 exact-law g12 nodes restricted to
    g12 >= t, walk fix, segment start at t, T_hi = 6), row-chunked like T388's wrapper; equivalence to the unchunked
    `est_seg` is checked on clean orig before the run (`t404_est.py check_k3e2 orig`).
  - **A-E2**: T340's variant A (`rare.est(variant="A")`, T317's transverse point, walk fix), the cheap variant that was
    "useful" on E2 for 3 clean models (T340) but one-signed low on fresh ones (T360/T363).
- **Truth (own code, `t404_truth.py`):** exact conditional-law MC given g12 >= t and g23 >= t: v2 ~ 90 phi(v)
  (1 - Phi(v + t)) Phi(v - t)^8, v1 ~ N | > v2 + t, the 8 others iid N | < v2 - t, random positions; 1e8 samples per t,
  through all 52 models. P(condition) by quadrature (agrees with 4M brute-force draws within 0.9-1.6 s.e. at all 3 t).
  Sanity: clean orig E2(.5) vs T340's GPU truth 9.38e-5. Diagnostic: hits by the rank the network picks.
- **Order:** estimates for both estimators sealed and pushed before any E2 truth is computed.
- **Go (fixed now, H-R1-style): >= 80% of the 144 noisy points within 2x, per estimator.** Also reported: median,
  one-signed bias, breakdown by sigma / base / t, F.
- **Expectation, written before running:** E2 is one-signed low on fresh clean models (T360: 76% within 2x, median .70),
  traced to the lows closure (T363: 56% of the deficit). T337 saw interior errors grow with noise. So the likely result
  is NO GO for both, worse as sigma grows if noise adds interior configurations the closure misses. Alternative: noise
  makes errors common and spread out (as in T371), and both estimators improve at large sigma.

## T404 part (a) results: GO (unverified; `python score_t404.py a` -> `results/t404_score_a.txt`, `results/t404_score_a.json`)

| | n | within 2x | median | F <= 1 | F <= 10 | median F |
|---|---|---|---|---|---|---|
| **all noisy** (go: >= 140) | 144 | **143** | 1.04 | 65 | 118 | 1.26 |
| sigma .001 | 36 | 36 | 1.07 | 26 | 36 | .38 |
| sigma .005 | 36 | 35 | 1.05 | 23 | 34 | .65 |
| sigma .02 | 36 | 36 | 1.03 | 8 | 26 | 3.1 |
| sigma .05 | 36 | 36 | 1.03 | 8 | 22 | 5.2 |
| clean | 12 | 12 | 1.05 | 11 | 12 | .04 |

- **The cut changes nothing that matters:** the largest move against T388 (T_hi = 6) is 6.8e-9, under the declared
  bound 7.04e-8. The one miss is T388's (orig@.005#0, t0 1.5, .36).
- **Cost:** 1.38-2.42M FPE per model (3 thresholds); **.63x T388's** (.61-.65) and **4.5x K3's** (4.3-4.8).
- **F (post-registration reading):** the rule beats sampling at equal compute (F <= 1) mostly where events are rare at its
  own cost (sigma <= .005: 49/72). At sigma .02-.05, p reaches 5e-5 to 5e-3 and sampling sees hundreds to thousands
  of events, so a 3-10% bias gives F > 1 (16/72 strict, 48/72 lenient).
- Note: the two lanes of part (a) were accidentally launched twice for lane 0. The estimator is deterministic, so this
  only duplicated work; all 52 entries were checked present.

## T404 part (b) results: E2 on the noisy copies: both GO by the pre-registered line, but carried by common events (unverified; `python score_t404.py b` -> `results/t404_score_b.txt`, `results/t404_score_b.json`)

Truths: 1e8 conditional samples per t; every point has >= 360 hits (rel. s.e. <= 5.3%). Clean orig E2(.5) = 9.40e-5
(T340 GPU: 9.38e-5). Truth range 1.4e-8 to 7.9e-2: at sigma .02-.05 the "interior" error is common (p up to 8%).

| estimator | C (FPE / model, 3 t) | noisy within 2x (go: >= 80%) | median | low / high | F <= 1 | F <= 10 |
|---|---|---|---|---|---|---|
| **K3-E2** | 1.55-2.81M | **139 / 144 (96.5%)** | .96 | 120 / 24 | 36 | 79 |
| **A-E2** | 11.5-28.6k | **128 / 144 (88.9%)** | .90 | 114 / 30 | **89** | **132** |

| subset | K3-E2 within 2x / median | A-E2 within 2x / median | A-E2 F <= 1 / <= 10 |
|---|---|---|---|
| sigma .001 / .005 / .02 / .05 | 32 / 35 / 36 / 36 (of 36) | 31 / 25 / 36 / 36 | 32 / 22 / 17 / 18 (strict) |
| **rare: truth <= 2e-4 (42 pts)** | **37 (88%) / .88** | **31 (74%) / .73** | 41 / 42 |
| common: truth > 2e-4 (102 pts) | 102 / .97 | 97 / .92 | 48 / 90 |
| clean bases (12) | 12 / .83 | 10 / .95 | 10 / 12 |

- **Misses are all at sigma <= .005 for both estimators.** K3-E2 (5): dev3_s526@.001#0 t .75 / 1.0 (.29, .22),
  dev3_s526@.001#1 t .75 / 1.0 (2.7, 3.5; truth 2.2e-6 / 7.5e-8), orig@.005#0 t 1.0 (.34). A-E2 (16): whole copies fail
  together and collapse as t rises (dev3_s526@.001#0 .095 -> .010, dev3_s528@.005#0 .22 -> .004, dev3_s528@.005#1
  .46 -> .027, orig@.005#2 .38 -> .13). That is the T360/T363 pattern (closure loses interior islands).
- **No single error type explains the misses.** The truth's mix of which rank the network picks (rank 1 / rank 3 /
  rank 4+) varies across the misses. Points where most errors pick rank 4+ are no worse (K3 64/64 within 2x).
- **Expectation vs result.** The pre-registered expectation (NO GO, worse with sigma) was wrong. Both estimators improve
  as sigma grows, as in T371: large noise makes the interior errors common and spread out, and the walks' closure
  captures them. The difficult regime is small noise, where the events are rare.
- **Equal compute.** A-E2's tiny cost makes it the one that beats sampling: F <= 1 on 89/144 (41/42 rare points),
  although it is the less accurate one. K3-E2's 2M FPE cost buys accuracy but loses to sampling wherever p C is large
  (F <= 1 on 36/144).

**T404 claims.**
1. [measured] (a) The graded g12 rule at T_hi = 3.5 is within 2x on 143/144 noisy points at .63x T388's cost
   (4.5x K3's). The pre-registered line (>= 140) is met. The cut moves estimates by <= 6.8e-9.
2. [measured] (b) On the interior event E2(t = .5, .75, 1) over the same noisy copies, K3-E2 is within 2x on 139/144 and
   A-E2 on 128/144. Both pass the pre-registered 80% line, both are one-signed low (median .96 / .90), and every miss is at
   sigma <= .005.
3. [measured] Restricted to rare points (truth <= 2e-4, 42 of 144; a post hoc split), K3-E2 is 37/42 within 2x (median .88)
   and A-E2 31/42 (74%, median .73). The pass in claim 2 is therefore carried largely by common events at large noise.
4. [measured] At equal compute A-E2 has F <= 1 on 89/144 and F <= 10 on 132/144; K3-E2 has 36 and 79.
5. [inferred] E2's known one-signed-low failure (T360) appears here only at small noise. Noise does not widen it; at
   sigma >= .02 it disappears because the events stop being rare.

**T404 alternatives / limits.** Part (a)'s truths were known (a code check). Part (b)'s truths were computed after the
estimates were sealed (dacaf3fd), but the estimators were already known to be one-signed low on E2. The panel is the same
4 bases and noise rays as T371 (copies are not independent across sigma). "Rare" vs "common" is a post hoc split. The
rare subset is 42 points from 4 bases, so its 74-88% within 2x rests on few copies (several misses are whole copies).
