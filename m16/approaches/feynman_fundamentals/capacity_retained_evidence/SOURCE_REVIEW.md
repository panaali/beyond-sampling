# T257 retained T234 evidence: source and schema findings

## Frozen scope

Claim8e739420/sharedc0246abb. Owner `claude/t192-gpu-cost-law` at `2ea6766d8fa0d30cc98348de247b9bec331545d1`. This commit adds capacity-study results even though its branch name refers to T192. Fifteen source/text/JSON blobs are copied exactly; six NPZ files are inventoried only by Git blob ID and size, without reading their payloads. `SCHEMA.json` records keys, types and list lengths, not numerical contractions. Prior T249 (`b9366ad3`) checked the same repaired executable source at d2fdf6bd; the intervening capacity-folder delta contains README and results/weights only. This review does not repeat that implementation gate or authenticate the old runs.

## Retained evidence versus current writers

1. `results/a30.json` retains a configuration dictionary, n_train/n_steps, wall_sec, a list of64 curve records with five loss/accuracy entries each, and five evaluation dictionaries. It has no `src_hash` field. Its evaluation dictionaries have n, seed, floating acc/SE and error-anatomy proportions; they contain neither `correct` nor `discordant_correct_vs`. The current `run.py` writes src_hash and the current `evaluate.py` writes both integer fields. Thus the retained JSON schema is compatible with the README's old-code description, not a record of the repaired writer's full output. The claimed later re-evaluation is not represented by those additional fields in this frozen file. This is a retained-evidence gap, not proof that re-evaluation did not occur elsewhere.

2. `results/zoo_code_seed0.json` retains seed, wall_sec and one evaluation dictionary with the same older field set. Current `repro_zoo.py` imports current `evaluate`, so a fresh successful evaluation would include the new integer fields. The old result lacks a source hash, package commit/version, checkpoint hash, device/library record and RNG state. No historical runtime identity follows merely from co-publication with current source.

3. The new six weight Git entries are content-addressed repository artifacts with names corresponding to five a30 seeds and one zoo-code seed. The JSON schemas do not bind their exact payload hashes. This phase does not decode shapes/dtypes/coefficients, compare a saved checkpoint with an evaluated model, or check their accuracy. A filename is not that binding.

4. The current reproducibility wrapper calls `alg_zoo.training.train("2nd_argmax", hidden_size=16, seq_len=10, seed=seed)` without explicit optimizer/schedule arguments, then loads a filename containing2^30 and evaluates a single model. This verifies the intended use of package defaults, conditional on the installed package. It does not freeze the historical package default implementation. Its training wall timer wraps `train`; final checkpoint loading, export and evaluation are outside. The current batched driver times its post-construction loop, also excluding final evaluation. These timers are not comparable end-to-end compute ledgers or Colab-unit totals.

## What the reproduction language establishes

The README's step-0 rule is that zoo-code and batched-trainer outcomes fall within a historical zoo seed spread. Passing that broad range is at most a coarse outcome sanity check. It does not establish identical trajectories, data streams, optimizer/CUDA semantics, mechanism, distribution of training outcomes or uninterrupted/resumed parity. A single zoo-code seed is not a paired multi-seed study. Different training data streams are explicitly acknowledged by the README. Earlier source repairs remain useful independently of these limits.

The printed spread endpoints and displayed accuracies are rounded. This source/schema phase does not treat rounded endpoints as exact acceptance thresholds or independently verify the table's arithmetic. The exact historical range and its intended rounding tolerance would be needed for a precise range test.

Training-seed variation, conditional evaluation MC uncertainty and best-seed/configuration selection remain separate. Current within-call discordance fields would enable paired comparisons of models evaluated together, but these retained old records omit them. Matching evaluation seed numbers alone does not supply missing paired correctness counts across the separate zoo-code and batched runs. No paired significance, architecture ceiling, no-material-improvement or equivalence conclusion is established here. The finite achieved-model result, if independently validated later, would supply a feasible-performance lower bound, not an upper bound on the architecture supremum.

The source's error anatomy uses the global second-largest input to define the blind stratum, then partitions nonblind errors by predicted input rank. The stored schemas expose proportions, not their integer joint event records. No anatomy sums, SEs, curve endpoints, model scores or actual error rates were recomputed. T201's blindness qualifications remain: a sign stratum is not proof that the network is rank-blind.

## Concrete prerequisites and alternatives

A later bounded saved arithmetic review could check displayed proportions, curve/end-point consistency and table transcription from these JSON files under a separate frozen scope. It would still not prove runtime or model identity. Stronger provenance would require the actual executed source/package/environment, checkpoint-to-evaluation mapping and retained integer paired outcomes. Actual CUDA interrupted/uninterrupted controls remain the separate T249 prerequisite; none are added by this result schema.

The missing fields may exist in untracked Colab/checkpoint or later evaluation artifacts. Old-code execution can be correct despite absent metadata. This review does not invalidate any reported empirical score, request training reruns, alter the owner's approved study budget or claim exhaustive evidence absence beyond the frozen subtree.
