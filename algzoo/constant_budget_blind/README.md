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

Pre-registered; no investigator has run yet.

## Results

None yet.

## Compute accounting

To report: sub-agent calls (by role) with tool calls and tokens from the Agent results; investigator probe
calls and sequences (from their scripts); scoring forward passes (shared scorer: 200,000 hidden + 200,000 full
forward passes per label file) and CPU time.

## Claims

None yet.

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

## Failure modes

Investigators may exceed the stated budget (the budget is instructed, not enforced; overruns are reported).
Text-only judges cannot check a label against the network. Five constants per neuron may be too few for
position-coded neurons regardless of understanding (E5/T7 sweep).
