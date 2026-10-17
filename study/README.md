# study/: Qwen3-0.6B blinded explainer/judge study (E1)

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md) · [PREREGISTRATION](../PREREGISTRATION.md) ·
[RESULTS, study 1](../RESULTS.md)

Built by [../build_items.py](../build_items.py) from activations collected by [../collect.py](../collect.py);
scored by [../analyze.py](../analyze.py) into [../results/summary.json](../results/summary.json).

| Path | Contents |
|---|---|
| [EXPLAINER_PROMPT.md](EXPLAINER_PROMPT.md), [JUDGE_PROMPT.md](JUDGE_PROMPT.md) | Prompt templates for the agents |
| [explain/](explain) | 10 blinded explainer packets (84 neuron poles + 28 random-direction controls, shuffled) |
| [explain_out/](explain_out) | Explainer outputs (explanation + category A/B/C/D) |
| [judge/](judge) | 10 judge packets (held-out snippets; 30 mismatched-explanation controls mixed in) |
| [judge_out/](judge_out) | Judge ratings 0-3 |
| [keys/](keys) | Answer keys (item -> neuron/direction, true activations). Agents were told not to read these. |

Review findings about this study (neuron-dominated controls, token overlap): R1 in the
[INDEX ledger](../INDEX.md#4-hypothesis-ledger) (H-Q2, H-Q3) and [../review/README.md](../review/README.md).
