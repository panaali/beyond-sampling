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

**Pre-registered; not started.** Next steps: pilot, then control choice, main collection, reproduction check,
items, explainers, judges, analysis.

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

## Results

(pending)

## Compute accounting

(pending: forward passes, sub-agent calls)

## Claims

(pending)

## Alternatives not ruled out

See the pre-registration above. Final list pending.

## Failure modes

- Memory: the fp32 model process peaks at about 3.9 GB RSS, above the 2 GB per-process rule. It runs as one
  process at a time (this container has no other agents), under `ulimit -v 8000000`.
- If the corpus reproduction check fails (tokenizer or library version drift), the items are not E1's units.
