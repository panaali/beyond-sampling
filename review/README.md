# review/: independent critical review

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md)

**R1** was a skeptical reviewer agent that re-ran analyses on the stored data. Its 10 findings and the corrections
they caused are recorded in the [INDEX registry (R1)](../INDEX.md#3-experiment-registry) and
[ledger](../INDEX.md#4-hypothesis-ledger) (H-Q2/3/4, H-A1, H-M3/4/5/6, H-E3/4), and applied to
[RESULTS.md](../RESULTS.md) and [m16/FINDINGS.md](../m16/FINDINGS.md).

| Script | Checks |
|---|---|
| [R1/q1.py](R1/q1.py) | H2 statistics (Fisher exact, batch-adjusted) |
| [R1/q2.py](R1/q2.py) | Stored activations: ranges and fp16 overflow check (none found) |
| [R1/q3.py](R1/q3.py), [R1/q3c.py](R1/q3c.py) | Random-direction controls decomposed into per-neuron contributions |
| [R1/q4.py](R1/q4.py) | Exact variance ratio of the stratified design |
| [R1/q5.py](R1/q5.py) | Judge token overlap / per-item AUROC |
