# essay_h2_rerun/: T6, the Qwen H2 re-run at a decisive size

↑ [Project README](../README.md) · [TASKS (T6)](../TASKS.md) · [RESULTS, study 1](../RESULTS.md) · [PREREGISTRATION](../PREREGISTRATION.md) · [review](../review/README.md) · [INDEX ledger H-Q1..Q4](../INDEX.md#4-hypothesis-ledger)

Owner: Claude child session of the DRI (T6), branch `claude/t6-essay-rerun`.

## Question

The Goodfire essay "Neural Networks Are Already Clean" says most neuron poles have "one main job". Our
pre-registered test (E1, [RESULTS.md](../RESULTS.md)) asked in H2 whether real neurons get that rating at least
30 percentage points more often than random directions in the same MLP space. If they don't, a "% one job"
figure does not show that *neurons* are clean. E1 measured +25 pp (CI +5 to +45) with 28 controls and one
explainer per item. Independent review (R1) judged this **inconclusive**: too few controls, and the random
directions were partly made of single massive neurons (median top-1 neuron share 29%).

T6 re-runs H2 at a size that can decide it. It uses dense controls that no single neuron dominates, 280
controls, two independent explainers per item, and n-gram deduplication before the judges.

## Pre-registration (written and pushed before any explainer or judge runs)

Everything in this section is fixed before the dense-control pilot, the main activation collection, and any
agent call. Later changes go in a dated *Amendments* list below, with the reason, and are reported as deviations.

### What stays the same as E1

- Model `Qwen/Qwen3-0.6B`, fp32, all 28 layers. A neuron is one coordinate of the input to `down_proj`
  (`silu(gate) * up`, 3072 per layer). A pole is its positive or negative side.
- Corpus: `NeelNanda/pile-10k`, 64-token windows, at most 4 windows per document, 18,750 windows (1.2M tokens).
  Positions 0-3 of each window are excluded. Window construction is copied from [../collect.py](../collect.py).
- The seed-0 draw of 32 neurons and 4 Gaussian directions per layer, so the neuron pool and the original
  controls are the same units as E1. **Reproduction check:** before any item is built, the stored activations
  must match E1's recorded values in [../study/keys/items.json](../study/keys/items.json) at the stored positions
  (all 112 E1 items, tolerance 1% of the item's max; this covers corpus, tokenizer and model). If it fails, I stop and report.
- Explainer packet per pole: 20 top examples (even ranks of the per-document top 40), 8 at 20-50% of max,
  6 at 5-20% of max. Same snippet format (24 tokens before, marked token, 4 after) and percent labels. Same
  explainer prompt ([../study/EXPLAINER_PROMPT.md](../study/EXPLAINER_PROMPT.md)) and categories A/B/C/D, with
  only the file paths changed.
- Judge per explanation: 12 held-out snippets (4 held-out top, 4 at 20-50%, 4 random positions), rated 0-3.
  Same judge prompt ([../study/JUDGE_PROMPT.md](../study/JUDGE_PROMPT.md)), paths changed. Judges never see
  activations. Mismatched-explanation judge items are mixed in as a control.
- Arms are shuffled and blinded together. Agents are not told that controls exist.

### What changes (the T6 design)

1. **Arms and sizes.** Per layer: 10 neurons drawn from the 32 (seed 11, random pole each), **10 dense
   control directions**, and 2 of the 4 original Gaussian directions (the E1 control type, kept to test R1's
   explanation). Totals: **280 neurons, 280 dense controls, 56 Gaussian controls = 616 items.**
2. **Dense controls ("top neurons projected out").** E1's controls inherited a few massive neurons' jobs.
   Candidate constructions, each a random Gaussian vector `g` reweighted per neuron and then normalised:
   - `G`: `g` (E1's control);
   - `S`: `g / sd_i` (every neuron gives the same variance);
   - `M`: `g / max_i |a_i|` (every neuron gives the same peak);
   - `PX`: `g` with the top 1% of neurons by `max|a_i|` set to zero (massive neurons projected out);
   - `SX`: `S` with the top 5% of neurons by `max|a_i| / sd_i` set to zero.

   `sd_i` and `max|a_i|` come from a **pilot** on the first 512 windows. The pilot uses statistics only:
   no text is read and no agent is involved. For each candidate the pilot draws 8 directions per layer (seed 10),
   takes each pole's top 40 pilot positions, and measures each position's **top-1 share** (the largest single
   neuron's contribution / the projection) and its top-5 share. **Selection rule:** use the non-`G` candidate
   with the lowest median top-1 share, provided that share is <= 0.10. If no candidate passes, I stop and amend
   this section before collecting. The main run then draws 12 fresh directions per layer (seed 20) from
   the chosen construction. The choice is written to `control_choice.json` and committed before collection.
   **Post-hoc validity check (pre-registered):** recompute the shares on the 12 strongest shown examples of
   every final dense-control item, with R1's method ([../review/R1/q3.py](../review/R1/q3.py)). Dense controls
   count as valid if the median top-1 share is <= 0.15 (E1's controls: 0.29). If they fail, H2r is reported as
   "control invalid", and an analysis restricted to items with share <= 0.15 is reported as exploratory only.
3. **Two independent explainers per item.** Every item appears in two packets, which come from two independent
   random partitions into packets of 14 items. Each packet goes to a fresh sub-agent (Agent tool, default
   model, never Fable). 616 x 2 / 14 = **88 explainer calls.**
4. **One judge per explanation.** 1,232 matched judge items plus 120 mismatched (an explanation paired with
   another item's snippets), in packets of 20: **68 judge calls.** Each judge is a fresh sub-agent.
   Total about 156 calls. Failed or malformed calls are re-run once. The hard cap is 300.
5. **n-gram deduplication before judging.** (a) A held-out snippet is dropped and replaced by the next candidate
   (from the odd ranks of the top 80, the 20-50% band pool, or fresh random positions) if it shares any 8-token
   n-gram with any of the unit's 34 shown examples or with an already kept held-out snippet. Near-duplicate
   documents in the Pile otherwise let a judge string-match. (b) Lexical overlap: I record, for every judged
   snippet, whether its marked token (stripped, lower-cased, length >= 2) appears verbatim in the explanation.
   H3 is reported on all snippets and on the lexical-free subset. R1 found 179/336 neuron top snippets were
   token lookups in E1. (c) Explanations are not edited.
6. **Blinding.** Items get random IDs. Answer keys stay in the git-ignored `data/t6/keys/` until every agent
   call is finished, then they are copied into this folder. Each agent is told to read only its packet.
7. **CIs.** A two-way bootstrap (2,000 resamples) over items and over explainer calls, within arm. This
   accounts for the clustering by explainer batch that E1's CIs ignored.

### Hypotheses and decision rule

Item A-score = the fraction of its explainers who rated it A (0, 1/2 or 1). Arm A-rate = the mean item
A-score.

- **H2r (primary, decides T6).** Delta = A-rate(neurons) - A-rate(dense controls), with its 95% CI, against
  E1's pre-registered 30 pp margin:
  - **supports** the privileged-basis reading if CI lower bound > +30 pp;
  - **refutes** it if CI upper bound < +30 pp;
  - **inconclusive** if the CI contains +30 pp, or if the dense controls fail the validity check.
- Secondary, reported in all cases:
  - **H2r-0.** Are neurons cleaner at all (CI lower bound > 0)?
  - **H2g.** Neurons minus Gaussian controls, a replication of E1's +25 pp, with the same margin rule.
  - **R1 test.** A-rate(Gaussian) - A-rate(dense). R1's explanation predicts > 0.
  - The baseline-adjusted clean share (A_n - A_d) / (1 - A_d).
  - The full category distributions.
  - Explainer agreement on A vs not-A: Cohen's kappa and raw agreement, per arm.
- **H1r.** Neuron A-rate: >= 60% supports, 40-60% inconclusive, < 40% counts against, at 0.6B (E1 bars).
- **H3r.** Neurons, matched explanations:
  - judge AUROC held-out top vs random >= 0.85;
  - mid (20-50%) vs random >= 0.70;
  - matched minus mismatched (top vs random) >= +0.15.

  All three are scored on all snippets and on the lexical-free subset. The pass/fail used for the overall
  verdict is the **lexical-free** one, because the all-snippet AUROC is known to be inflated by token
  lookup. This is a deviation from E1, chosen in advance.
- **Overall verdict**, using E1's rule shape:
  - **supported at small scale** if H1r, H2r and H3r all pass;
  - **refuted at small scale** if H2r refutes or H3r fails;
  - **inconclusive** if H2r is inconclusive;
  - **mixed** otherwise (for example H1r alone fails, which the essay's small-model caveat allows).

### Power calculation ([power.py](power.py), [power.json](power.json))

The simulation draws per-item latent A-probabilities from a Beta distribution with concentration 2 or 10, and
adds per-call logit shifts with sd 0.4 or 0.8. The control A-rate is 20%.

- **s.e. of Delta ~= 0.028-0.032**, compared with **0.107 for E1's design** (84 vs 28 items, one explainer).
- Probability of each verdict, by the true Delta (kappa = 2, tau = 0.4):

  | true Delta | 0 | 10 pp | 20 pp | 25 pp | 30 pp | 35 pp | 40 pp | 50 pp |
  |---|---|---|---|---|---|---|---|---|
  | P(refute) | 1.00 | 1.00 | 0.91 | 0.38 | 0.03 | 0 | 0 | 0 |
  | P(support) | 0 | 0 | 0 | 0 | 0.02 | 0.33 | 0.88 | 1.00 |
  | P(inconclusive) | 0 | 0 | 0.09 | 0.62 | 0.95 | 0.67 | 0.12 | 0 |
  | P(CI excludes 0) | 0.03 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

So the design decides H2r unless the true Delta lies within about +-8 pp of the 30 pp bar. That is an
indifference zone of about 22-38 pp, compared with roughly 0-50 pp for E1's design. If the true Delta is
near E1's +25 pp, an inconclusive result is still likely (62%). The CI then pins Delta to about +-6 pp, which
is reported as the main quantitative answer. No sample size within the 300-call cap removes an indifference zone
at a fixed margin.

### Alternatives to keep in view (AGENTS rule 3)

- Dense controls may be rated messy because their top examples are less extreme or more diffuse, not
  because the neuron basis is privileged. Check: compare arms on the LLM-free top-token share and on judge
  AUROC.
- Explainers may detect the "texture" of control packets and rate them differently for that reason, not
  because they see more jobs. This cannot be fully excluded. I report the per-arm distribution of shown
  examples' percent labels.
- The explainer model differs from E1's agents (E1's model is not recorded). A-rate levels may shift. Delta
  is a within-run comparison, so it is less affected.

## Status

**Done (2026-10-19). Pre-registered verdict: INCONCLUSIVE.** The only reason is that the dense controls narrowly
failed the pre-registered validity gate: median top-1 neuron share 0.163, against the 0.15 bar. The primary
contrast itself clears the 30-pp bar: Delta = +43.8 pp, 95% CI +33.3 to +53.7. Exploratory checks, set out in
advance as exploratory only, suggest the gate failure does not drive the result (see Results). This is
unverified. The DRI decides whether to accept it, and an independent re-check is needed (AGENTS rule 4).

## Amendments

**Amendment 1 (2026-10-18, after the pilot, before the main collection; no text or agent involved).** The pilot
(first 512 windows, 30,720 tokens) gave the following median top-1 shares: `G` 0.232, `S` 0.175, `M` **0.115**,
`PX` 0.196, `SX` 0.161 (top-5 shares 0.59, 0.51, 0.40, 0.54, 0.49). None reached the pre-registered 0.10 bar,
so per the rule I stopped before collecting. Changes:

- **New candidates.** Three are added: `MX5`, `MX10`, `MX20` = `M` with the top 5%, 10% or 20% of neurons by
  `max|a_i| / sd_i` (the most kurtotic) projected out. They are scored on the same pilot with the same seed-10
  directions.
- **New selection rule.** Candidates are taken in order of how much they remove: `S`, `M`, `PX`, `SX`, `MX5`,
  `MX10`, `MX20`. The first one with median top-1 share <= 0.10 is used. If none passes, the candidate with the
  lowest share is used, provided it is <= 0.15 (the post-hoc validity bar). Otherwise I stop. Order of
  intrusiveness is preferred over lowest share because projecting out more neurons builds the control from a
  less typical set of neurons.
- **New confound check (secondary, H2r-c).** An `MX` control is made only of the less kurtotic neurons. If those
  neurons are themselves less "clean", that alone would lower the control's A-rate. So I also report
  Delta for the subset of sampled neurons that the chosen construction does *not* project out in their layer.
  This needs no new agent calls.

**Amendment 1 outcome (pilot, before collection).** `MX5` 0.1139, `MX10` 0.1151, `MX20` 0.1161 (top-5 shares
all about 0.40). Projecting out kurtotic neurons does not lower the share below `M`'s 0.1147; the floor is
about 0.11. None is <= 0.10, so the fallback applies and the lowest-share candidate is used: **`MX5`**, which
is within noise of `M`. See [control_choice.json](control_choice.json). It projects out 153 of 3,072
neurons per layer, so H2r-c applies.

## Results

**Reproduce:**
- `python essay_h2_rerun/analyze_t6.py essay_h2_rerun/keys` gives [summary.json](summary.json). It also needs
  `data/t6/pilot.json` (a copy is in `keys/pilot_stats.json`) and `data/t6/neuron_idx.npy`, both regenerated by
  `collect_t6.py`.
- Full pipeline:
  1. `collect_t6.py pilot`
  2. `collect_t6.py main`
  3. `build_t6.py check`
  4. `build_t6.py explain`
  5. 88 explainer sub-agents
  6. `build_t6.py judge`
  7. 68 judge sub-agents
  8. `density_check.py`
  9. `analyze_t6.py`

**Reproduction of E1** ([repro_check.json](repro_check.json)): all 5,110 activations recorded in E1's keys are
reproduced exactly (worst error 0.0000 of max). Corpus, tokenizer, model and the seed-0 units are identical to E1.

**Control validity** ([density_check.json](density_check.json); R1 method, 12 strongest shown examples per item):

| arm | median top-1 neuron share | median top-5 share | items with top-1 <= 0.15 |
|---|---|---|---|
| dense (MX5) | **0.163** | 0.47 | 47% |
| Gaussian (E1's control) | 0.334 | 0.67 | 25% |

The dense controls halve the single-neuron dominance but miss the pre-registered 0.15 bar, so the validity gate
**fails**. The direction recomputed from the full activations matches the stored projections to within 0.05% of max.

**Explainer categories** (2 explainers per item; CIs from a two-way bootstrap over items and explainer calls):

| arm | n | A-rate (95% CI) | A | B | C | D | A vs not-A agreement (kappa) |
|---|---|---|---|---|---|---|---|
| neurons | 280 | **64.6%** (56.2-72.3) | 64.6 | 14.1 | 18.4 | 2.9 | 0.89 (0.77) |
| dense controls | 280 | **20.9%** (14.8-27.4) | 20.9 | 8.9 | 61.1 | 9.1 | 0.92 (0.75) |
| Gaussian controls | 56 | 33.0% (18.0-49.5) | 33.0 | 17.0 | 48.2 | 1.8 | 0.88 (0.72) |

**Hypotheses:**
- **H2r (primary).** Delta(neurons - dense) = **+43.8 pp, CI +33.3 to +53.7**. The CI lies entirely above +30. On
  its own this would mean "supports". The validity gate failed, so the pre-registered verdict is **inconclusive
  (control invalid)**.
- **H2r-0.** Neurons are cleaner than dense controls: yes (CI excludes 0).
- **H2r-c** (confound check, amendment 1). Restricted to the 265/280 sampled neurons that MX5 does not project out:
  Delta = +43.8 pp, items-only CI +36.7 to +50.7. Unchanged.
- **H2g** (replication of E1's contrast). Delta(neurons - Gaussian) = +31.6 pp, CI +13.4 to +48.4: inconclusive,
  as in E1 (E1 found +25, CI +5 to +45).
- **R1 test.** A(Gaussian) - A(dense) = +12.1 pp, CI -4.5 to +29.7. The direction R1 predicted, but not significant
  (n = 56 Gaussian).
- **Baseline-adjusted clean share** (A_n - A_d)/(1 - A_d) = 55% (CI 43-66%).
- **H1r.** Neuron A-rate 64.6% (CI 56-72): **supports** (>= 60%), borderline as in E1 (61%).
- **H3r** (neurons, matched explanations; pass/fail on the lexical-free subset): **pass**.

  | measure | all snippets | lexical-free | bar |
  |---|---|---|---|
  | top vs random | 0.935 (0.927-0.943) | **0.919** (0.909-0.930) | >= 0.85 |
  | mid vs random | 0.800 (0.786-0.815) | **0.778** (0.762-0.794) | >= 0.70 |
  | matched - mismatched (top vs random) | +0.396 | **+0.386** | >= +0.15 |
  | dense controls: top / mid vs random | 0.808 / 0.570 | 0.777 / 0.569 | (descriptive) |
  | Gaussian controls: top / mid vs random | 0.846 / 0.521 | 0.820 / 0.503 | (descriptive) |

  - 33% of neuron held-out top snippets have their marked token verbatim in the explanation. E1 had 53%; the
    n-gram dedupe and the measure differ.
  - 8-gram dedupe replaced at least one held-out top snippet for 12% of neuron items and 19-21% of control items.
- **Pre-registered overall verdict: INCONCLUSIVE**, because H2r is inconclusive (control invalid).

**Exploratory** (pre-registered as exploratory only; not part of the verdict):
- Dense controls with top-1 share <= 0.15 (n = 132): A-rate 25.4%. Delta(neurons - these) = +39.3 pp,
  items-only CI +30.1 to +47.9.
- Within the dense arm, A-score does not track single-neuron dominance:
  - correlation(top-1 share, A-score) = **+0.01**;
  - A-rate by top-1-share tercile: 26%, 18%, 18%.

  In the Gaussian arm it does: correlation +0.20, terciles 24%, 36%, 40%. That matches R1's finding for E1. So
  among dense controls, the residual dominance (0.163 vs 0.15) does not appear to make controls look cleaner.
  Removing the most dominated controls *raises* the control A-rate slightly (21% to 25%) instead of lowering it.

**LLM-free descriptives** (medians):

| measure | neurons | dense | Gaussian |
|---|---|---|---|
| top-token share of top 80 | 0.025 | 0.013 | 0.025 |
| activation mass above 20% of max | 6.5% | 32% | 37% |

### Deviations from the original E1 protocol (complete list)

1. **Sizes:** 280 neurons, 280 dense and 56 Gaussian controls (E1: 84 neurons and 28 Gaussian). 10 neurons per
   layer drawn from the same seed-0 pool of 32 (seed 11; it may include E1's units).
2. **New control type:** max-normalised Gaussian directions with the 5% most kurtotic neurons projected out (MX5).
   It was chosen from 8 candidates by a mechanical pilot rule; amendment 1 changed that rule after no candidate
   reached 0.10. Only 2 of E1's 4 Gaussian directions per layer were kept.
3. **Two explainers per item** (E1: one). The two explainers saw two independent random packings of 14 items per
   packet (E1: about 11). Each explainer was a fresh sub-agent on the session default model. E1's agent model is
   not recorded, so the model may differ. One explainer call (r2_p24) omitted an item and was re-run once in full,
   per the pre-registered rule. The failed output is kept in `explain_failed/`.
4. **One judge per explanation:** 1,232 matched and 120 mismatched items, 20 per packet (E1: 112 + 30, about 14 per
   packet). The packing kept two items with the same snippets out of the same packet.
5. **8-gram dedupe of held-out snippets.** Candidates came from the top 80 (E1: top 40), the band pool and fresh
   random positions. Explanations were not edited. 72 judge items have only 8-9 snippets because the 20-50% band
   ran out (E1 behaved the same way).
6. **H3 pass/fail on the lexical-free subset** (E1: all snippets). Both are reported; both pass.
7. **CIs:** two-way bootstrap over items and explainer calls (E1: items only).
8. **Decision rule for H2:** CI against the 30-pp margin (E1: point estimate), plus a control-validity gate (new).
9. **Prompt texts:** E1's, with only paths and ID formats changed ([EXPLAINER_PROMPT.md](EXPLAINER_PROMPT.md),
   [JUDGE_PROMPT.md](JUDGE_PROMPT.md)). The judge prompt adds "some items have fewer than 12 snippets".
10. **Storage:** activations stored per layer as (unit, token) float16 (E1: (window, pos, layer, unit) float16).
    Values are identical (reproduction check).
11. **Blinding:** keys were kept in git-ignored `data/` during all agent calls (E1: `study/keys/` in the repo, with
    agents told not to read it). An audit of all 157 agent transcripts found that every agent opened only its own
    packet and its own output file.

## Compute accounting

- **Model forward passes** (Qwen3-0.6B fp32, CPU):
  - pilot: 2 x 512 windows;
  - main: 18,750 windows of 64 tokens (1.2M tokens);
  - density check: 336 x 12 windows.

  In all, about 20,800 window-forwards, about 1.5 CPU-hours on 4 threads. Peak RSS was about 4 GB in one process
  (above the 2 GB rule, as declared).
- **Sub-agent calls: 157 of the 300 cap.**
  - 89 explainer calls (88 + 1 re-run), about 80k tokens each.
  - 68 judge calls, about 68k tokens each.
  - No Fable.
- **Agent wall time:** about 30-50 s per call, run in waves of 10-17.
- **Analysis:** two-way bootstrap with 2,000 resamples, a few CPU-minutes.

## Claims

- [measured] With the same model, corpus, units and protocol as E1, at 280 vs 280 items with 2 explainers, neurons
  are rated "one main job" 64.6% of the time and MX5 dense controls 20.9%. Delta = +43.8 pp (CI +33.3 to +53.7).
- [measured] The dense controls fail the pre-registered validity gate (median top-1 share 0.163 > 0.15). The
  pre-registered verdict is therefore **inconclusive**.
- [measured] E1-style Gaussian controls are rated A 33% of the time; E1 found 36%. Their Delta vs neurons
  (+31.6 pp, CI +13 to +48) is again inconclusive against the 30-pp bar.
- [measured] H1r: 64.6% of neuron poles are rated A (supports, borderline). H3r: explanations predict held-out
  activations (lexical-free AUROC 0.92 top, 0.78 mid; mismatched 0.53).
- [inferred] The gate failure is unlikely to explain the large Delta. Among dense controls, A-rate does not rise
  with single-neuron dominance (correlation +0.01). The least-dominated half gives Delta = +39 pp, with an
  items-only CI lower bound of +30.1. If the gate is waived, the result supports E1's H2 (the neuron basis is
  privileged by the essay's own rating method). This needs the DRI's decision and an independent check.
- [inferred] R1's explanation (random directions inherit single neurons' jobs) is supported directionally for
  Gaussian controls, which become cleaner as dominance rises. The direct contrast Gaussian vs dense is not
  significant.
- [speculative] Explainers may partly detect control "texture": dense controls carry 32% of their mass above 20%
  of max, neurons 6.5%. That would inflate Delta without the basis being privileged in the essay's sense.

## Alternatives not ruled out

- **Texture, not jobs.** Dense-control packets differ systematically: flatter activation profiles and more
  varied top tokens. Explainers may down-rate them for looking "mixed" even where a coherent feature exists. The
  judges' lower AUROC for dense explanations (top 0.78 vs 0.92) fits both readings: either the explanations are
  worse because there is no single job, or the units are harder to describe. **Not checked:** an arm of neurons
  matched to dense controls on activation-profile statistics.
- **Construction artefact.** MX5 directions are a particular kind of dense mixture: max-normalised, with
  kurtotic neurons removed. Other dense directions (for example PCA or ICA axes, or SAE-like features) could be
  rated cleaner. Not tested.
- **Same-family raters.** All explainers and judges are Claude models, like the essay's tools. Different rater
  families were not tried.
- **Model scale.** 0.6B only. Says nothing direct about Qwen3-32B.
- **The gate itself.** 0.15 was a judgement call. Its failure by 0.013 is within the pilot-to-full-corpus
  increase (0.114 pilot vs 0.163 full) that the pilot could not foresee.

## Failure modes

- Memory: the fp32 model process peaks at about 3.9 GB RSS, above the 2 GB per-process rule. It runs as one
  process at a time (this container has no other agents), under `ulimit -v 8000000`.
- If the corpus reproduction check fails (tokenizer or library version drift), the items are not E1's units.
