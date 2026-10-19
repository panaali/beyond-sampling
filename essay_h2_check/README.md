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

Pre-registered (072a5e78), amended once before collection; pass 1 (stats) running.

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

(pending)

## Compute accounting

(pending)

## Claims

(pending)

## Alternatives not ruled out

(pending; see pre-registration)

## Failure modes

- The fp32 model process needs about 4 GB. This is above the 2 GB rule, waived by the DRI for this one process in
  this container; only one model process runs at a time.
- Stored activations take about 9 GB in `data/t383/`.
