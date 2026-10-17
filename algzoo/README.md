# algzoo/: pre-registered legibility vs understanding test (E3-E5)

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md) ·
[Pre-registration (addendum E4)](../PREREGISTRATION.md) · [RESULTS, study 2](../RESULTS.md)

A blinded investigator agent labelled each hidden neuron and wrote a formula `f(i, x)` (no weights, at most 5
constants per neuron). [score.py](score.py) scores the formulas on held-out inputs: per-neuron R^2 and the
accuracy of the network rebuilt from the formulas.

| File | Model | Result |
|---|---|---|
| [labels_4_3.py](labels_4_3.py) | M_4,3 (32 params, positive control, E3) | 4/4 legible, 100% of gap recovered ([../results/algzoo_4_3.json](../results/algzoo_4_3.json)) |
| [labels.py](labels.py) | M_16,10 (432 params, E4) | 14/16 legible, rebuild 54% ([../results/algzoo_16_10.json](../results/algzoo_16_10.json)); caveat: ledger H-A1 |
| [labels_32_10.py](labels_32_10.py) | M_32,10 (1,376 params, E5) | Draft, unscored (stopped by the DRI) |

Command: `python algzoo/score.py algzoo/labels_4_3.py --zoo 4,3` (default model: the 432-param example).
