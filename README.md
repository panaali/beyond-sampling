# clean-neurons

An independent test of two claims about neural-network interpretability:

1. Goodfire's essay *"Neural Networks Are Already Clean"* (Cammarata, Oct 2026): most MLP neurons have one main job.
2. ARC's AlgZoo challenge (Hilton): legible neurons are not understanding. Calculate the accuracy of a 432-parameter
   RNN from its mechanism, at least as precisely as random sampling at equal compute.

## Start here

| Document | What it is |
|---|---|
| [TASKS.md](TASKS.md) | **Task board: what is in progress (and who holds it) and the prioritized backlog. Claim a task here before starting work** |
| [STATUS.md](STATUS.md) | **Current status**, in plain words plus a technical summary |
| [INDEX.md](INDEX.md) | Single source of truth: questions, repo map, **experiment registry** (E1-E27, LB2, R1), **hypothesis ledger** (every claim with its status and evidence), synthesis, open questions |
| [RESULTS.md](RESULTS.md) | Write-up of the Qwen3-0.6B neuron study and the AlgZoo legibility study, with review corrections |
| [m16/FINDINGS.md](m16/FINDINGS.md) | How the 432-param RNN M_16,10 computes the 2nd argmax: mechanism, error anatomy, estimators, failed hypotheses |
| [PREREGISTRATION.md](PREREGISTRATION.md) | Hypotheses and thresholds committed before any data was collected |
| [AGENTS.md](AGENTS.md) | Rules for all agents (Claude, Codex, humans): claiming tasks, README template, skepticism, independence. ([CLAUDE.md](CLAUDE.md) only imports it for Claude Code) |

## Folders

| Folder | Contents |
|---|---|
| [m16/](m16/README.md) | Analysis of M_16,10: shared code, ARC-protocol harness, leaderboard |
| [m16/approaches/](m16/approaches/README.md) | 18 estimator attempts at ARC's challenge, one folder per agent |
| [algzoo/](algzoo/README.md) | Pre-registered legibility test (E3-E5): investigator labels + held-out scorer |
| [lit/](lit/README.md) | Literature scans by era and by field |
| [results/](results/README.md) | Machine-readable results (leaderboards, truths, study summaries) |
| [review/](review/README.md) | Independent critical review (R1) scripts |
| [study/](study/README.md) | Qwen study: blinded explainer/judge packets, agent outputs, answer keys |
| [tools/](tools/README.md) | Doc tooling (link checker, linkifier) |

Top-level scripts belong to the Qwen study pipeline: [collect.py](collect.py) (activations) →
[build_items.py](build_items.py) (blinded packets) → [analyze.py](analyze.py) (pre-registered scoring);
[population.py](population.py) (LLM-free statistics).

## Reproduce

```
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install transformers pyarrow numpy scipy
pip install -e /path/to/alg-zoo            # https://github.com/alignment-research-center/alg-zoo
python m16/leaderboard.py truths && python m16/leaderboard.py run sampling stratified control_variate
python tools/check_links.py                # verify doc links
```
