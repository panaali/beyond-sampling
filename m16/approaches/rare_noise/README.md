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
| T371's misses (go: >= 12) | 16 | 0 | **16** | .21 | .89 |
| T371's hits (go: <= 6 broken) | 128 | 128 | **127 (1 broken)** | 1.03 | 1.05 |
| all noisy | 144 | 128 | **143** | 1.02 | 1.04 |
| truth tie share < 50% | 107 | 107 | 107 | 1.03 | 1.05 |
| tie share 50-80% | 12 | 10 | 11 | .97 | 1.08 |
| tie share >= 80% | 25 | 11 | **25** | .38 | .96 |
| clean bases (reference) | 12 | 8 | **12** | 1.01 | 1.07 |

- **Misses, K3 -> graded:** dev3_s528 copies .002-.37 -> .82-1.06 (12/12, e.g. @.001#2 t0 1.5 .0015 -> .99). orig
  copies .38-.47 -> .52-.88 (4/4; 2 of them only just inside, .52 and .57).
- **The one broken hit:** orig@.005#0, t0 1.5 (tie share 57%): .79 -> .36.
- **No overshoot anywhere** (max graded ratio 1.66, clean orig t0 1.0). T367's s547 pattern (2.1-2.4x high at a large
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
