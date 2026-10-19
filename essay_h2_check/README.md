# essay_h2_check/: T383, an independent resolution of T6 (the essay's "one main job" claim, H2)

↑ [Project README](../README.md) · [TASKS (T383)](../TASKS.md) · [INDEX ledger H-Q1..Q4](../INDEX.md#4-hypothesis-ledger) · [RESULTS](../RESULTS.md) · [review](../review/README.md)

Owner: Claude child session of the DRI (T383), branch `claude/t383-essay-check`. A different agent from T6.

## Question

Goodfire's essay "Neural Networks Are Already Clean" says most neurons have "one main job". The project's H2 asks
whether real neurons get that rating at least **30 percentage points** more often than random directions in the same
MLP space. T6 (branch `claude/t6-essay-rerun`, commit `acd53054`) measured +43.8 pp (CI +33.3 to +53.7) but was
**inconclusive** because its dense controls failed the pre-registered validity gate (median top-1 neuron share
0.163 > 0.15). T6 also flagged an unexamined alternative: dense controls have a different *activation texture*
(32% of their mass above 20% of max, against 6.5% for neurons), which explainers might rate as "messy" for that
reason alone.

T383 resolves both points with its own code:
1. a control construction that passes the 0.15 gate, **checked on the final items before any agent call**;
2. a second neuron arm **matched to the controls on texture**, next to the plain neuron arm.

## Pre-registration (written and pushed before any activation collection or agent call)

Everything below is fixed before collection. Later changes go in a dated *Amendments* list, with reasons, and are
reported as deviations. A short smoke test of the code on <= 64 windows (code check only, output discarded) may
follow this push.

### Kept from T6 / E1

- Model `Qwen/Qwen3-0.6B`, fp32, all 28 layers. A neuron is one coordinate of the input to `down_proj`
  (`silu(gate) * up`, 3,072 per layer). A pole is its positive or negative side. A direction `w` has pole values
  `s * (h . w)`.
- Corpus as E1: `NeelNanda/pile-10k`, 64-token windows, at most 4 per document, 18,750 windows; positions 0-3
  excluded. (Re-implemented in [collect.py](collect.py).)
- Explainer packet per pole: 20 top examples (even ranks of the per-document top 40), 8 at 20-50% of max, 6 at
  5-20% of max; snippet = 24 tokens before, marked token, 4 after; percent labels. Judge item: 4 held-out top (odd
  ranks), 4 held-out 20-50%, 4 random positions, rated 0-3. 8-gram dedupe of held-out snippets against the 34 shown
  examples and earlier kept ones. Lexical flag (marked token, stripped, lower-cased, length >= 2, verbatim in the
  explanation). Mismatched-explanation judge items as a control.
- Prompts: T6's [EXPLAINER_PROMPT.md](EXPLAINER_PROMPT.md) and [JUDGE_PROMPT.md](JUDGE_PROMPT.md), with only paths
  changed. Two independent explainers per item (two random packings), one judge per explanation. Every agent is a
  fresh sub-agent (Agent tool, default model, **never Fable**), told to read only its packet. Arms are shuffled and
  blinded together; agents are not told that controls exist. Answer keys stay in git-ignored `data/t383/keys/` until
  all agent calls are done.
- Item A-score = fraction of its 2 explainers rating A. Arm A-rate = mean item A-score. CIs: two-way bootstrap over
  items (within arm) and explainer calls, 2,000 resamples, percentile 95%. Re-run a failed or malformed call once.

### New in T383

**1. Arms (11 per layer, 308 each, 924 items).**
- `plain`: 11 neurons per layer drawn uniformly from the live neurons (seed 383), random pole.
- `control`: 11 dense directions per layer from the construction chosen in step 2.
- `matched`: 11 neurons per layer matched to that layer's 11 controls on texture (step 3).

**2. Control construction and the density gate.** The gate statistic is T6's: for an item, the median over its 12
strongest *shown* examples of `max_i(s w_i a_i) / (s w . a)`; the arm statistic is the median over items. T6 picked
its construction on a 512-window pilot, whose per-neuron maxima underestimate the full-corpus maxima; the share then
rose from 0.114 (pilot) to 0.163 (final). T383 removes that gap:
- **Pass 1** ([collect.py](collect.py) `stats`) computes every neuron's positive and negative maximum (`M+_i`,
  `M-_i`) on the **full corpus**.
- Candidate constructions, in this fixed order (least to most changed from a Gaussian draw):
  - `Mf`: `g_i / max(M+_i, M-_i)`, g Gaussian (T6's `M`, with full-corpus maxima);
  - `Rf`: `r_i / max(M+_i, M-_i)`, r random signs (removes Gaussian magnitude spread);
  - `Rp`: `r_i / M^(s r_i)_i`, pole-aware, so that every neuron's largest contribution to the item's pole is exactly
    1 (`M^(+)` = `M+`, `M^(-)` = `M-`).

  Dead neurons (max < 1e-6) get weight 0. Each construction gets 19 directions per layer with a pole drawn per
  direction: directions 0-7 are a **selection half**, 8-18 are the **item half**.
- **Pass 2** (`main`) records, for every candidate direction and window, the maximum pole value, its position and
  the exact top-1 / top-5 neuron shares there (fp32, from the hidden state).
- **Selection rule** ([build.py](build.py) `select`): compute the gate statistic on the selection half (8 x 28 =
  224 directions per construction). Take the first construction in the order above with median <= **0.13** (a
  margin below the gate). If none, take the lowest one provided it is <= 0.15. Otherwise stop and amend.
- **Gate (pre-registered, binding):** recompute the shares by re-forwarding the 12 strongest shown examples of
  every one of the 308 final controls (R1/T6 method). **If the median exceeds 0.15, no agent call is made**: I stop
  and amend. Controls therefore cannot fail the gate after the fact.

**3. Texture-matched neuron arm.** Texture = T6's `frac_mass_above_20`: over all non-excluded tokens of the pole,
(sum of values above 20% of the pole max) / (sum of positive values).
- Pass 1 also builds per-neuron log-binned histograms (160 bins, 1e-4 to 1e4), giving an approximate texture for
  every neuron pole.
- **Pool** (as amended, see Amendment 1): per layer, excluding the plain neurons, 64 eligible neuron poles spread over
  approximate texture 0.10-0.60. Pass 2 stores them.
- **Matching rule:** per layer, an optimal 1:1 assignment (Hungarian, minimum total |texture difference|, exact
  textures from the stored activations) of the 11 controls to 11 distinct pool neurons.
- **Balance check (before agents):** pass if |median texture(matched) - median texture(controls)| <= 0.03 and at
  least 80% of pairs are within a caliper of 0.05.
  - If the check fails, the arm is still run as built. H2m is then computed on the within-caliper pairs only (paired
    contrast), and is called *untestable* if fewer than 100 pairs remain.
- Secondary balance, reported without a bar: `flat20` (the 20th shown top example as a fraction of max, the
  visible part of texture) and the top-token share.

**4. Sizes and calls.** 924 items x 2 explainers = 1,848 explanations, in packets of 21: **88 explainer calls**.
1,848 matched + 120 mismatched judge items, in packets of 24: **82 judge calls**. 170 in total, plus re-runs; hard
cap 200. Packets are larger than T6's (14 / 20) to fit the 200-call cap with 3 arms (deviation).

### Hypotheses and decision rule

- **H2c (primary, the T6 question with valid controls).** Delta_c = A(plain) - A(control).
- **H2m (texture-matched).** Delta_m = A(matched) - A(control).

Each uses the same rule: **supports** if the CI lower bound > +30 pp; **refutes** if the CI upper bound < +30 pp;
**inconclusive** otherwise.

**Overall T383 verdict on H2:**

| H2c | H2m | verdict |
|---|---|---|
| supports | supports | **SUPPORTED**: neurons are > 30 pp cleaner than valid dense controls, also at matched texture |
| supports | refutes | **TEXTURE-DEPENDENT**: the plain margin does not survive texture matching |
| supports | inconclusive or untestable | **SUPPORTED (plain); texture question open** |
| refutes | any | **REFUTED** |
| inconclusive | supports | **INCONCLUSIVE (plain); matched supports** |
| inconclusive | other | **INCONCLUSIVE**; the CIs are the answer |

Secondary, reported in all cases:
- A(plain) - A(matched), the texture effect among neurons;
- the paired matched-control contrast;
- H2-0: the CIs of Delta_c and Delta_m against 0;
- H1: A(plain) >= 60% supports, 40-60% inconclusive, < 40% against;
- H3, for plain and matched separately, lexical-free subset binding, with T6's bars: AUROC top vs random >= 0.85,
  mid vs random >= 0.70, matched minus mismatched >= +0.15;
- category distributions; explainer agreement (kappa) per arm;
- exploratory: logistic fit of A-score on texture within neurons (plain + matched).

### Power check ([power_check.py](power_check.py), [power.json](power.json); own simulation)

308 items per arm and 2 explainers in packets of 21. Item A-probabilities ~ Beta(kappa) around the arm mean; logit
shift ~ N(0, tau) per explainer call; control rate 20%; same two-way bootstrap as the analysis (400 resamples,
300 simulations per cell). The true Delta is set on the latent item probabilities; call noise pulls realised rates
towards 50%, so the realised Delta is slightly smaller, most visibly at tau = 0.8.

- **s.e. of Delta ~= 0.037-0.042** in all scenarios. T6's own CI (+33 to +54) implies s.e. ~0.052 at 280 items.
- Probability of each decision, by true Delta (kappa = 2, tau = 0.4; the other two scenarios are in `power.json`):

  | true Delta | 0 | 10 pp | 20 pp | 25 pp | 30 pp | 35 pp | 40 pp | 44 pp | 50 pp |
  |---|---|---|---|---|---|---|---|---|---|
  | P(refute) | 1.00 | 1.00 | 0.76 | 0.15 | 0 | 0 | 0 | 0 | 0 |
  | P(support) | 0 | 0 | 0 | 0 | 0 | 0.16 | 0.67 | 0.97 | 1.00 |
  | P(inconclusive) | 0 | 0 | 0.24 | 0.85 | 1.00 | 0.84 | 0.33 | 0.03 | 0 |
  | P(CI excludes 0) | 0.02 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

  At T6's point estimate (44 pp) P(support) is 0.97 (0.92 at tau = 0.8, 0.79 at kappa = 10, tau = 0.8). The
  indifference zone is roughly 22-40 pp. If texture matching halves the gap to ~22 pp, H2m most likely refutes.
  No design within the 200-call cap removes the indifference zone around a fixed margin.

### Alternatives to keep in view (AGENTS rule 3)

- **Texture, not jobs.** The matched arm tests this. If it is only partly matched, the residual gap is reported.
- **Construction artefact.** All candidates are max-normalised random mixtures. A low-share control is a mixture of
  many neurons near their peaks. That is what a valid dense control means here, but other dense families (PCA, ICA,
  SAE features) are not tested.
- **Pool neurons are not random.** Matched neurons are selected for high texture (dense, less sparse neurons). They
  may differ from typical neurons in other ways (layer position, polysemanticity). A(plain) - A(matched) is reported
  for that reason.
- **Same-family raters:** all agents are Claude models.

## Status

**Done (2026-10-19). Pre-registered verdict: SUPPORTED (plain); texture question open.**
- **H2c supports.** Neurons beat dense controls that pass the gate by +44.6 pp, CI +36.3 to +52.9.
- **H2m is inconclusive.** Texture-matched neurons beat the same controls by +37.0 pp, CI +28.7 to +45.5. The lower
  bound falls 1.3 pp short of the +30 bar.

This is unverified. Nobody confirms their own claim (AGENTS rule 4); the DRI decides.

## Amendments

**Amendment 1 (2026-10-19, after the code smoke test, before any collection).** The smoke test (48 windows, output
discarded) showed that "highest approximate texture" favours degenerate poles that fire on a handful of tokens, all
near their max (texture ~1.0). Such a pool would hold few neurons near the controls' expected texture (~0.3 in T6).
The pool rule (step 3) is replaced by:
- **Eligible poles** have at least 100 tokens above 20% of the pole max (approximate, from new count histograms
  in pass 1).
- Per layer, excluding plain neurons, each neuron contributes its higher-texture eligible pole.
- **Pool:** for 64 targets evenly spaced over texture 0.10-0.60, take the nearest unused eligible pole (64 per layer).

The matching rule, balance check and everything else are unchanged.

## Results

**Reproduce:**
- From the committed keys: `python essay_h2_check/score.py essay_h2_check/keys` gives [summary.json](summary.json)
  (2,000-resample bootstrap; about 90 s).
- Full pipeline:
  1. `collect.py stats`
  2. `collect.py main`
  3. `build.py select` (writes [choice.json](choice.json))
  4. `build.py explain`
  5. 88 explainer sub-agents
  6. `build.py judge`
  7. 82 judge sub-agents
  8. `score.py`

  Prompts are [EXPLAINER_PROMPT.md](EXPLAINER_PROMPT.md) and [JUDGE_PROMPT.md](JUDGE_PROMPT.md); packets are in
  `explain/` and `judge/`, outputs in `explain_out/` and `judge_out/`.

**Gate and balance, checked before any agent call** ([choice.json](choice.json)):

| check | value | bar | result |
|---|---|---|---|
| selection half: median top-1 share, `Mf` / `Rf` / `Rp` | 0.106 / 0.064 / 0.057 | first <= 0.13 in order | **`Mf` chosen** |
| final 308 controls: median top-1 share (re-forwarded, 12 strongest shown examples) | **0.104** (top-5: 0.37; 87% of items <= 0.15) | <= 0.15 | **pass** (T6: 0.163) |
| re-forwarded vs stored projection, worst relative error | 0.05% | - | consistent |
| texture median, matched vs controls | 0.488 vs 0.487 (gap 0.001) | gap <= 0.03 | **pass** |
| pairs within the 0.05 caliper | 301/308 (98%) | >= 80% | **pass** |
| `flat20` median (visible texture, not matched on) | plain 0.54, matched 0.77, controls 0.76 | - | matched also balanced |

The selected `Mf` is T6's `M` construction with full-corpus maxima instead of pilot maxima. That alone moves the
final share from T6's 0.163 to 0.104. This supports T6's own diagnosis that the pilot maxima caused the gate miss.

**Explainer categories** (2 explainers per item; CIs from a two-way bootstrap over items and explainer calls):

| arm | n | A-rate (95% CI) | A | B | C | D | A vs not-A agreement (kappa) |
|---|---|---|---|---|---|---|---|
| plain neurons | 308 | **65.3%** (59.1-71.2) | 65.3 | 15.1 | 16.9 | 2.8 | 0.88 (0.74) |
| texture-matched neurons | 308 | **57.6%** (51.3-63.9) | 57.6 | 6.7 | 26.6 | 9.1 | 0.91 (0.82) |
| dense controls (`Mf`) | 308 | **20.6%** (15.3-26.3) | 20.6 | 8.1 | 56.8 | 14.4 | 0.90 (0.69) |

**Hypotheses:**
- **H2c (primary).** Delta = A(plain) - A(control) = **+44.6 pp, CI +36.3 to +52.9**: **supports**. This is T6's
  +43.8 pp almost exactly, now with controls that pass the gate.
- **H2m.** Delta = A(matched) - A(control) = **+37.0 pp, CI +28.7 to +45.5**: **inconclusive**. The CI contains +30.
  - The paired version gives the same picture: all 308 pairs +37.0 (CI +28.0 to +46.2); the 301 within-caliper pairs
    +36.2 (CI +26.9 to +44.8).
  - **H2-0:** both contrasts exclude 0 by a wide margin.
- **Texture effect among neurons.** A(plain) - A(matched) = +7.6 pp, CI -1.3 to +16.6. Matching texture removes
  about a sixth of the gap at most; the rest remains.
- **H1.** A(plain) = 65.3%: **supports** (>= 60%), as in E1 (61%) and T6 (65%).
- **H3** (lexical-free, binding): **passes** for plain and for matched neurons.

  | arm | top vs random | mid vs random | matched - mismatched |
  |---|---|---|---|
  | plain | 0.886 (0.874-0.898) | 0.733 (0.717-0.750) | +0.333 |
  | matched | 0.860 (0.843-0.874) | 0.720 (0.703-0.740) | +0.307 |
  | controls (descriptive) | 0.762 | 0.549 | +0.210 |
  | bar | >= 0.85 | >= 0.70 | >= +0.15 |

  - Mismatched explanations score top vs random 0.553.
  - All-snippet values are higher (plain 0.914 / 0.765).
  - 33% of held-out top snippets have their marked token verbatim in the explanation.
- **Pre-registered overall verdict: SUPPORTED (plain); texture question open** (decision table, H2c supports and
  H2m inconclusive).

**Exploratory** (not part of the verdict):
- Within neurons (plain + matched), the logistic slope of A on texture is -0.17 per SD (SD 0.22): a weak decline.
- Within controls, correlation(texture, A-score) = -0.32. Denser controls are rated less clean, so texture matters
  *within* an arm. At equal texture, neurons still beat controls by about 37 pp.

### Deviations from T6's protocol (complete list)

1. **Arms:**
   - 308 plain neurons drawn uniformly from all live neurons (seed 383). T6: 280 from E1's seed-0 pool of 32 per layer.
   - New: 308 texture-matched neurons.
   - 308 `Mf` dense controls (T6: 280 `MX5`).
   - No Gaussian (E1-style) arm.
2. **Control construction:** max-normalised Gaussian with **full-corpus** maxima. It was chosen on a held-out
   selection half, and the 0.15 gate was checked on the final items before any agent call. T6 used pilot maxima and
   checked the gate after the agents.
3. **Packing:** 21 items per explainer packet (T6: 14) and 24 per judge packet (T6: 20), to fit three arms under the
   200-call cap. 120 mismatched judge items.
4. **Judge prompt:** "followed by 12 numbered text snippets" became "up to 12 numbered text snippets (some items have
   fewer; give one rating per snippet)". 27 items had fewer than 4 held-out mid snippets. T6 made the same change.
5. **Re-runs:** one explainer call (r1_p04) omitted an item and was re-run once in full, per the rule. The failed
   output is kept in `explain_failed/`.
6. **Blinding audit:** I checked all 171 agent transcripts.
   - Every agent read only its own packet and wrote only its own output file.
   - One explainer (r1_p03) used `sed` and a one-line `python3 -c` on its own output file to fix a JSON quoting
     error, which breaks "do not run code". It read nothing else.
   - Keys stayed in git-ignored `data/t383/keys/` until all calls were done; they are now copied to `keys/`.
7. **Code:** collection, item building, scoring and the power simulation are new code written for T383. The
   corpus-window definition, packet format and prompts follow E1/T6 by design.
8. **Amendment 1** (pool rule), made before collection.

## Compute accounting

- **Model forward passes** (Qwen3-0.6B fp32, CPU, 4 threads):
  - pass 1: 18,750 windows of 64 tokens (about 75 min);
  - pass 2: 18,750 windows (about 65 min);
  - gate re-forward: about 3,400 windows;
  - smoke tests: 2 x 48 windows.

  In all, about 41,000 window-forwards, about 2.5 h wall time. Peak RSS was about 4.1 GB in one process (the waived
  limit).
- **Storage:** 9.5 GB in `data/t383/` (132 units x 28 layers x 1.2M tokens, float16, plus window-max records).
- **Sub-agent calls: 171 of the ~200 cap.**
  - 89 explainer calls (88 + 1 re-run), about 95k tokens each.
  - 82 judge calls, about 71k tokens each.
  - Default model; no Fable.
- **Analysis:** two-way bootstrap with 2,000 resamples; H3 bootstrap with 500 resamples over judge items; about 1.5 CPU-min.

## Claims

- [measured] With controls that pass the pre-registered density gate (median top-1 share 0.104 <= 0.15, measured on
  the final items before any agent call), neurons are rated "one main job" 65.3% of the time against 20.6% for dense
  controls. Delta = +44.6 pp, CI +36.3 to +52.9, above the 30-pp bar. H2c **supports**.
- [measured] Neurons matched to the controls on texture (median mass above 20% of max 0.488 vs 0.487) are rated A
  57.6%. Delta vs controls = +37.0 pp, CI +28.7 to +45.5. H2m is **inconclusive** against the 30-pp bar; it is far
  above 0.
- [measured] Texture explains little of the gap: A(plain) - A(matched) = +7.6 pp, CI -1.3 to +16.6.
- [measured] H1 supports (65.3%). H3 passes for plain and matched neurons (lexical-free top/mid AUROC 0.89/0.73 and
  0.86/0.72).
- [inferred] T6's gate miss came from pilot-estimated neuron maxima. With full-corpus maxima the same `M`-type
  construction passes, and the contrast is unchanged (+44.6 vs +43.8).
- [inferred] T6's "texture, not jobs" alternative is mostly ruled out. At equal activation texture, neurons are still
  rated A about 37 pp more often than random dense directions. Whether that matched margin exceeds the 30-pp bar is
  not settled at this sample size.
- [speculative] The matched margin's point estimate (+37) suggests a design about 1.5 times this size would resolve
  H2m. That is not pre-registered here.

## Alternatives not ruled out

- **Other texture features.** Matching was on mass above 20% of max; `flat20` turned out balanced too. Other visible
  cues remain unmatched: top-token diversity (controls 0.013 vs neurons 0.025 top-token share) and the variety of
  contexts among top examples. A control with mixed jobs will naturally show more varied tokens, so this cue is
  partly the effect being measured.
- **Matched neurons are a selected set.** They are the dense, high-texture neurons (eligible poles with at least 100
  tokens above 20% of max). They are rated A less often than plain neurons and get more D ratings (9% vs 3%). They may
  differ from typical neurons in ways besides texture.
- **Construction family.** All controls are max-normalised Gaussian mixtures. PCA, ICA or SAE directions were not
  tested.
- **Same-family raters.** All explainers and judges are Claude models, like the essay's tools.
- **Scale.** 0.6B only.
- **Bigger packets** (21 vs 14 items) might change how explainers calibrate their ratings. All three arms share the
  same packets, so the contrasts are less exposed than the levels.

## Failure modes

- Memory: the fp32 model process needs about 4 GB (above the 2 GB rule, waived for this container). The first
  launch, under `ulimit -v 8000000`, failed at model load; it ran with `ulimit -v 30000000`. Memmaps over the 9.5 GB
  stores also exceed a tight `ulimit -v`.
- `select` needs the full window-max records. If pass 2 is interrupted, it resumes from `winmax.npz`.
- If no construction passes the gate on the final items, the pipeline stops before any agent call, by design.
