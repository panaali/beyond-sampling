# T24: fresh blinded investigators and category judges on an equal-effort model-size ladder

↑ [Project README](../../README.md) · [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · [E5 / T7](../constant_budget/README.md) · [E4 pre-registration](../../PREREGISTRATION.md)

## Question

The essay "Neural Networks Are Already Clean" claims that naming what each neuron does is a workable route to
understanding a network. E5 (`algzoo/constant_budget/`) wrote a neuron-by-neuron account of M_32,10 whose
frozen formulas recovered 88.54% of the chance-to-model gap (criterion 90%), but its categories were completed
retrospectively by a different agent, so it is not a clean test. **With the same investigation effort per
model, blinded investigators and blinded judges, does the quality of a neuron-by-neuron account (label
agreement between independent investigators, and reconstruction of the network from the labels' formulas)
hold up as the model grows from 8 to 16 to 32 hidden neurons?**

## Method (pre-registered 2026-10-18, before any investigator ran)

### Models (the ladder)

Three AlgZoo 2nd-argmax RNNs at sequence length T = 10, each loaded with `zoo_2nd_argmax(H, 10)` (AlgZoo's
default best-of-5 seed), hidden sizes **H = 8, 16, 32**. Accuracies on the shared scorer's held-out set
(200,000 Gaussian sequences, seed 20261016), measured during setup:

| Model | Params | Model accuracy | Replacement accuracy needed for 90% of the gap |
|---|---:|---:|---:|
| M_8,10 | 152 | 42.674% | 39.406% |
| M_16,10 (zoo default; not the `example_2nd_argmax` challenge model) | 432 | 94.952% | 86.456% |
| M_32,10 (same model as E5) | 1,376 | 99.442% | 90.498% |

M_8,10 is a weak model (43%), so the ladder confounds size with model quality; see *Alternatives*. Nothing
else from these models (activations, weights, neuron statistics) was looked at before freezing this plan.

### Investigators (the essay's method)

- **3 independent investigators per model, 9 in total**, each a fresh sub-agent (Agent tool, Claude Opus;
  never Fable) given the identical prompt [prompts/investigator.md](prompts/investigator.md) with only H, its
  directory and the budget substituted.
- **Equal effort per model:** every investigator gets the same budget of **80 tool calls**, whatever H. Per
  neuron effort therefore halves at each step up the ladder; this is the intended "constant budget per model"
  design (does a fixed-effort neuron account scale?). Actual tool calls and tokens are recorded from the Agent
  results.
- **Access:** only `probe.hidden(x)` (final hidden state h_10 for inputs the investigator chooses) and
  `probe.predict(x)` ([probe_template.py](probe_template.py)), as in E4/E5. No weights, no recurrence, no W_oh,
  so no investigator can compute replacement accuracy; they can only fit per-neuron activations.
- **Blinding:** each investigator works in its own directory outside the repository (session scratchpad,
  random 4-letter code), is told not to open any file outside it, and never sees other investigators' or other
  sizes' results, E4/E5 labels, INDEX or this README. After each run I audit its scripts for weight access
  (`alg_zoo`, `torch`, `weight`, `state_dict`, `.rnn`) and its notes.md file list for reads outside its
  directory. A violating run is excluded and rerun once with a fresh code; both are reported.
- **Formula rule (as E4):** at most 5 fitted real constants per neuron; searched discrete choices allowed but
  declared in `STRUCT`. A neuron declaring >5 constants is scored but flagged non-compliant.
- All 9 investigators run concurrently (light CPU: OMP_NUM_THREADS=1, `ulimit -v 2500000`).

### Frozen label categories

- **Legibility category** (the E4 rubric, unchanged): A = one main job with related offshoots; B = two related
  jobs; C = several unrelated jobs; D = not understood.
- **Role taxonomy** (frozen here, before any label exists; the judge chooses one primary role and optionally a
  secondary one): `MAX`, `SECOND`, `GAP`, `POS_MAX`, `POS_SECOND`, `LAST`, `RAW`, `OTHER`. Definitions are in
  [prompts/judge_role.md](prompts/judge_role.md), which is the exact judge prompt.

### Judges (blind to size and investigator)

Label texts are extracted from each `labels.py`. A fixed regex removes any mention of a neuron index or
neuron count (`neuron \d+`, `\d+ neurons`, `h\[\d+\]`, `h_10\[\d+\]`); every redaction is logged. Items get
random IDs; items from all three sizes are pooled and shuffled (numpy seed 2410) into batches of 28. Judges see
only label text, never the formula, R², size, investigator or neuron index.

1. **Role judges** ([prompts/judge_role.md](prompts/judge_role.md)): every label (8+16+32 = 56 neurons × 3
   investigators = 168 labels; 6 batches) is judged by **2 independent judge calls** (the second sees the
   batch in reversed order): category A–D, primary role, secondary role. 12 calls.
2. **Pair judges** ([prompts/judge_pair.md](prompts/judge_pair.md)): for each neuron and each pair of its
   3 investigators, a **matched pair** (both labels of the same neuron; 56 × 3 = 168 pairs). For each matched
   pair (investigators a, b, neuron i), a **control pair** (a's label of neuron i, b's label of neuron π(i),
   π a random derangement of the same model's neurons, seed 2410): 168 controls. P/Q order is randomised.
   336 items, 12 batches, each judged by **2 independent calls**: 24 calls. Verdict scores SAME = 1,
   PARTIAL = 0.5, DIFFERENT = 0, averaged over the two judges.

Judges are sub-agents told to use no tools. A malformed or incomplete answer is re-asked once with the same
batch; still-missing items are reported as missing (never imputed).

### Scoring rules (frozen)

Per model size H:

- **Reconstruction (primary):** the unchanged shared scorer `algzoo/score.py <labels.py> --zoo H,10`
  (200,000 held-out sequences, seed 20261016) for each investigator: replacement accuracy, **gap recovered**
  = (repl − 0.1)/(model − 0.1), and per-neuron R². G_H = median over the 3 investigators of gap recovered.
  An investigator's account **passes** if gap recovered ≥ 0.90.
- **Label agreement (primary):** S_H(matched) and S_H(control) = mean pair-judge score over matched / control
  pairs; **specificity D_H = S_H(matched) − S_H(control)** (how much more two independent labels of the same
  neuron agree than labels of different neurons of the same model). 95% CI by bootstrap over neurons
  (10,000 resamples, seed 2410; a resampled neuron carries its 3 matched pairs and their 3 controls).
- **Role agreement (secondary):** Cohen's κ between investigators of the judge-assigned primary role on the
  same neuron, averaged over the 3 investigator pairs and the 2 judges; same-role rate for matched vs control.
- **Legibility (secondary, E4 criterion):** fraction of neurons with judge category A/B and R² ≥ 0.8,
  averaged over the 2 judges and 3 investigators; criterion ≥ 75%. Self-rated categories reported separately.
- **Judge reliability:** inter-judge κ on roles and categories, and agreement on pair verdicts.

**Verdicts (fixed in advance):**

- *Reconstruction holds with size* if G_32 ≥ 0.90 and G_32 ≥ G_8 − 0.05; *degrades* if G_32 < 0.90 and
  G_32 < G_8 − 0.05; otherwise *mixed*.
- *Labels hold with size* if D_32 ≥ D_8 − 0.15 and the 95% CI of D_32 excludes 0; *degrade* if
  D_32 < D_8 − 0.15; otherwise *mixed*.
- **Overall:** the neuron account's quality **holds up** with model size if both hold; **does not hold up** if
  either degrades; **mixed/inconclusive** otherwise. M_16,10 is reported as the intermediate rung and checked
  for monotonicity but does not enter the verdict.
- Secondary comparison: the 3 fresh M_32,10 accounts against E5's 88.54%.

### Independent scoring

After all judging, a separate sub-agent that has not seen any of the above outputs writes its own scorer from
scratch (own code, a different held-out seed, 200,000 sequences), recomputes replacement accuracy, gap
recovered and R² for all 9 label files, and recomputes the agreement statistics from the raw judge outputs.
Discrepancies are reported, not reconciled silently.

### Sub-agent budget

9 investigators + 12 role-judge calls + 24 pair-judge calls + 1 independent scorer = **46 calls**, plus at most
20 reserved for the re-asks and reruns defined above. Hard cap 200.

### Key files

- [probe_template.py](probe_template.py): the investigator's only model interface (H substituted).
- [prompts/](prompts/): the exact investigator and judge prompts.
- `runs/<code>/`: each investigator's deliverables and scripts, copied after the run (added later).
- `judging/`, `results/`, analysis scripts: added after the runs; they implement the rules above.

## Protocol decisions logged during the run (before any score was computed)

- 2026-10-19, `wfpn` (H=8): the investigator disclosed (i) zero-padding inputs to read the hidden state after
  shorter prefixes (allowed: any inputs), (ii) regressing h_10 on the 9-step prefix state plus x9, i.e.
  *estimating* the one-step map (close to the weights) from probe outputs, used for understanding only and in
  no formula, and (iii) reading its own background-script output file in the harness task folder (no
  blinding leak). Decision: not a violation of the written rules (no weights read; the audit regex targets
  direct access), so the run is kept, flagged, and every per-size result is also reported excluding `wfpn`.

## Status

Done (2026-10-19). All 9 investigators and 36 judge calls completed. No judgments were missing or invalid, and
no run was excluded. Independent rescoring: see the section below. Results are supported, not confirmed
(rule 4); the DRI decides registration.

## Results

**Pre-registered verdict: mixed/inconclusive.** Reconstruction is *mixed*: G_32 = 0.824 < 0.90, but not
below G_8 − 0.05 = 0.535. Labels *hold*: D_32 = 0.885 ≥ D_8 − 0.15 = 0.506, and the CI excludes 0. Leaving
out the flagged run `wfpn` (pre-declared sensitivity check) gives the same verdicts: G_8 = 0.572, D_8 = 0.813.

### Reconstruction (shared scorer, 200,000 held-out sequences, seed 20261016)

| Model | Run | Replacement acc. | Model acc. | Gap recovered | Passes 90%? | Neurons with R² ≥ 0.8 |
|---|---|---:|---:|---:|:--:|---:|
| M_8,10 | agsc | 29.11% | 42.67% | 0.585 | no | 2/8 |
| M_8,10 | gcap | 28.28% | 42.67% | 0.560 | no | 3/8 |
| M_8,10 | wfpn | 29.69% | 42.67% | 0.603 | no | 2/8 |
| M_16,10 | hnrt | 35.90% | 94.95% | 0.305 | no | 15/16 |
| M_16,10 | wduk | 38.42% | 94.95% | 0.335 | no | 15/16 |
| M_16,10 | yyuc | 40.64% | 94.95% | 0.361 | no | 14/16 |
| M_32,10 | dncx | 83.70% | 99.44% | 0.824 | no | 26/32 |
| M_32,10 | krvc | 79.61% | 99.44% | 0.778 | no | 24/32 |
| M_32,10 | sfbp | 92.97% | 99.44% | **0.928** | **yes** | 25/32 |

Median gap recovered: **G_8 = 0.585, G_16 = 0.335, G_32 = 0.824**. This is not monotone in size. One fresh
M_32,10 account (`sfbp`) passes the 90% criterion; E5's 88.54% sits inside the fresh range (0.778–0.928).

**Single-neuron sensitivity** (`acc_replacing_one_neuron` in `results/score_*.json`): replacing one true
neuron with its formula costs on average **5.2 points** of accuracy at H = 8, **12.3 points** at H = 16
(a single neuron with R² 0.97 drops M_16,10 from 95% to 76%), and **0.26 points** at H = 32. The H = 16
failure therefore comes from how sensitive the readout is to small activation errors, not from poor labels.
That model's neurons have the highest legibility and agreement on the ladder.

### Labels (blinded judges)

| Model | S matched | S control | **D** (95% CI) | Role κ between investigators | Same role: matched / control | Legible, judge cats (self cats) |
|---|---:|---:|---:|---:|---:|---:|
| M_8,10 | 0.917 | 0.260 | **0.656** (0.427–0.854) | 0.36 | 0.54 / 0.21 | 29% (29%) |
| M_16,10 | 0.953 | 0.240 | **0.714** (0.583–0.833) | 0.66 | 0.73 / 0.17 | 84% (81%) |
| M_32,10 | 0.979 | 0.094 | **0.885** (0.831–0.935) | 0.83 | 0.86 / 0.15 | 72% (74%) |

Independent investigators' labels for the same neuron were judged to describe the same function almost
always, at every size. Labels for different neurons of the same model were rarely judged the same. The
specificity D *increases* with size, mainly because the control score falls: the larger model's neurons are
more distinct from one another. S matched is near ceiling, which compresses D. Legibility by the E4
criterion (A/B and R² ≥ 0.8 for ≥ 75% of neurons) passes only at H = 16. At H = 32 it reaches 72% (just
below), and at H = 8 it is low (29%), limited by R², not by category.

Judge reliability: inter-judge κ on roles 0.91, on categories 0.75; pair verdicts agree exactly on 95.2%
(κ 0.92). Judge role distributions differ by size (H = 8: mostly OTHER/POS_SECOND/MAX; H = 16 and 32: LAST, MAX,
GAP, POS_MAX), so the H = 8 model was described in a different vocabulary ("shortfall below earlier top
values").

### Reproduce (from the repository root, `source .venv/bin/activate`)

```bash
python algzoo/constant_budget_blind/collect.py <scratch>/t24 <scratch>/t24_assign.json   # audit (needs the scratch dirs; outputs are committed)
OMP_NUM_THREADS=1 python algzoo/constant_budget_blind/score_runs.py        # reconstruction table
python algzoo/constant_budget_blind/build_judging.py                       # judge batches (seed 2410; regenerates key.json)
python algzoo/constant_budget_blind/analyze.py                             # labels table + verdicts -> results/summary.json
cd algzoo/constant_budget_blind && python sensitivity.py                   # wfpn exclusion
```

### Audit

All 9 runs delivered complete `labels.py` files: no declared count above 5 constants, none of the banned
strings. The regex hits were benign: the word "weight" in prose, `yyuc/check.py` searching for the banned
strings, and outside paths only in my own TASK.md. No run read weights. `wfpn` estimated the one-step map
by regression (logged above, kept). Several runs (`wfpn`, `gcap`, `dncx`) zero-padded inputs to see
shorter-prefix states, which the rules allow. Self-reported file lists show no reads of other folders except
`wfpn`'s own task-output file. Budget: every investigator stayed under 80 tool calls (harness counts 23–39).

## Compute accounting

| Sub-agent role | Calls | Tool calls per call | Tokens per call | Wall time |
|---|---:|---:|---:|---:|
| Investigators | 9 | 23–39 (total 292) | 115k–148k (total 1.17M) | 13–28 min each, run concurrently |
| Role judges | 12 | 3 | ~55k | ~20 s |
| Pair judges | 24 | 3 | ~57k | ~20–25 s |
| Independent rescorer | 1 | see below | see below | see below |

**Total: 46 sub-agent calls** (cap 200). 4 further launches were refused by the harness's concurrency limit
and never ran; I re-launched them later. No Fable. Scoring: 9 shared-scorer runs × (200,000 hidden + 200,000
full forward passes + H × 200,000 single-neuron replacements). Investigator probe calls are in their scripts
under `runs/` (large `.npy`/`.npz` caches were not copied). All CPU, under 2 GB per process.

## Claims

- [measured] With an equal 80-tool-call budget per model, no investigator's account of M_8,10 or M_16,10
  recovers 90% of the chance-to-model gap; 1 of 3 fresh M_32,10 accounts does (0.928). Medians: 0.585 /
  0.335 / 0.824.
- [measured] The fresh blinded M_32,10 accounts span 0.778–0.928 gap recovered. E5's retrospective 88.54% is
  within that range, so E5's near-miss is typical of what a blinded investigator gets on this model.
- [measured] Independent investigators' labels of the same neuron are judged equivalent far above a
  same-model derangement control at every size, and the margin grows with size (D = 0.66 / 0.71 / 0.89).
  Judges agree with each other strongly (κ 0.91–0.92).
- [measured] Replacing one neuron by its formula costs ~12 accuracy points on M_16,10 but ~0.3 on M_32,10.
- [inferred] Label quality (agreement, legibility) and reconstruction come apart. M_16,10 has the most
  legible and highly agreed labels but the worst reconstruction. Naming neurons consistently, even with
  R² > 0.9, does not mean the account reproduces the network. Whether it does depends on how sensitive the
  readout is to activation errors, which differs between models.
- [inferred] On this ladder, the neuron account's quality does not degrade from 8 to 32 neurons, but the
  pre-registered verdict is mixed/inconclusive: reconstruction at 32 is below 90% for the median account.
  The non-monotone reconstruction means size is not the dominant variable here.
- [speculative] M_32,10's redundancy (many partly overlapping neurons, small single-neuron effects) is why its
  accounts reconstruct well, so a neuron-by-neuron account may work best on larger, more redundant models
  rather than worse. Separating this from model accuracy needs more seeds per size.

## Alternatives not ruled out (declared before running)

1. **Model quality confound:** M_8,10 is only 43% accurate; its neurons may implement a cruder (or messier)
   algorithm, so differences between rungs may reflect training outcome rather than size. Cheapest check
   (not in this run): repeat with other AlgZoo seeds or at T where all sizes are accurate.
2. **Budget per neuron vs. size:** a constant per-model budget halves per-neuron effort per rung, so a decline
   could mean "needs more effort", not "cannot be done". Cheapest separating test: an equal-per-neuron arm
   (budget ∝ H) on M_32,10; not in this run.
3. **Judge leniency / shared vocabulary:** generic labels ("encodes the gap") can look the same across
   neurons; the derangement control measures this, but only within a model.
4. **Investigator homogeneity:** all investigators are the same model family with the same prompt, so their
   agreement may reflect shared priors rather than the network. Agreement is not correctness;
   reconstruction is the check against the network.
5. **n = 3 investigators per size** gives a coarse median; the reconstruction verdict could flip with more.

Added after the results:

6. **Readout sensitivity, not size:** the single-neuron sensitivity differs 50-fold between M_16,10 and
   M_32,10, which can explain the non-monotone reconstruction on its own. Test: other AlgZoo seeds per size
   (e.g. `zoo_2nd_argmax(16, 10, seed=k)`, plus the `example_2nd_argmax` model), recording each one's
   single-neuron sensitivity.
7. **Same-model-family agreement:** every investigator and judge is the same model family, so high agreement
   may reflect shared descriptive habits. The 16-neuron case shows agreement can be high while reconstruction
   fails, so agreement must not be read as correctness.
8. **Judge ceiling:** S matched ≈ 0.92–0.98 leaves little room, so D mainly tracks the control score.
   A stricter judge rubric (e.g. requiring position sets to match) was not tried.
9. **Under-used budget:** investigators used 23–39 of 80 calls, so "equal budget" means an equal cap, not
   equal effort spent.

## Failure modes

Investigators may exceed the stated budget (the budget is instructed, not enforced; overruns are reported).
Text-only judges cannot check a label against the network. Five constants per neuron may be too few for
position-coded neurons regardless of understanding (E5/T7 sweep). Gap recovered is a whole-network
quantity: a fragile readout (M_16,10) can turn small per-neuron errors into large accuracy losses, so it
penalises accurate accounts of fragile models. The investigators could not see the readout, so their overall
"how the network computes the answer" pictures are untested against it (several said so explicitly).
