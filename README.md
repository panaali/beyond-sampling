# Beyond Sampling

**Can we predict a neural network’s mistakes by understanding how it works?**

ARC’s challenge uses a tiny network with 16 neurons and 432 adjustable numbers, called weights. It reads 10 numbers and tries to identify the second-largest. The goal is to work out how often it gets the answer right from its weights and internal calculations, using about as much computation as random testing would need for the same precision.

Original sources: [ARC’s AlgZoo challenge](https://www.alignmentforum.org/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) · [AlgZoo models and code](https://github.com/alignment-research-center/alg-zoo).

**[Interactive Tutorial](https://claude.ai/artifact/22zmMZh9WU6LAPLWqJmXRX)** · [Results in plain language](INDEX.md) · [Current status](STATUS.md)

## Predict, explain, repair, verify

The bigger goal is to catch mistakes too rare for ordinary testing, understand why they happen, and find changes that reduce them. Here are the main findings so far, on small networks with precisely defined tasks:

| Step | What we have shown | What we still need to watch for |
|---|---|---|
| **Predict rare mistakes** | On some tested cases, we estimated failure probabilities as small as **one in billions** without needing billions of ordinary trials. | The methods can miss other failure patterns; success on one case does not establish reliability everywhere. |
| **Explain the cause** | We traced a studied rare mistake to the balance between internal comparisons and predicted how small weight changes would affect its frequency. Independent interventions checked those predictions. | This is a tested explanation of a particular failure mechanism, not a complete account of every mistake. |
| **Repair the behavior** | Changing just **12 weights** made one rare mistake **17–19 times less likely**. A separate implementation reproduced the result. | Another type of mistake became more common. A targeted repair is not automatically an overall safety improvement. |
| **Verify the evidence** | Independent checks caught a numerical error in a proposed probability bound and showed how an apparently reassuring uncertainty warning could miss failure modes. A local numerical repair and a conservative reporting alternative were checked. | The repairs still need to be integrated and validated in the full trained-network workflow. We do not yet have a complete mathematical safety guarantee. |

These are separate, scoped demonstrations—not yet one end-to-end safety system. The [results ledger](INDEX.md) explains the evidence and limitations; the detailed records are H-R1, H-R2, H-R3, T462 and T463.

**A concrete win on the smaller challenge: our method outperformed ARC’s two published estimators in our reproduction.** ARC described its methods for the four-neuron, three-number network as roughly matching random testing. Our mechanism-derived method did substantially better: about **50 times lower squared error than sampling** under the tested computation comparison. An independent check confirmed our advantage over sampling. This result applies to the studied trained network and specified weight changes; it does not establish a solution to the full challenge or a wall-clock speedup. [Published-method comparison](m16/approaches/arc_m43_repro/README.md) · [Independent check](review/understand_m43/README.md).

## Two candidates for ARC’s full challenge

We have **two candidates for meeting the challenge**. Both have encouraging results on previously unused networks trained for the same task. Neither has been accepted by ARC, and each leaves a different question unanswered.

- **More accurate than random testing at the same compute budget.** We follow the network’s calculations exactly along selected slices of possible inputs, then combine the answers. On a fresh test panel, this produced about **half the squared error** of random testing. The open question: we have not proved how accurately those slices represent all the other inputs. That matters because ARC asks for an estimate grounded in understanding the network.
- **A second candidate tested at a much smaller budget.** It uses no random testing while it runs: we focus on the situations where the largest input numbers are close together and use a fixed set of simplifying assumptions. The estimate achieved about **2.6 times the squared error** of random testing at the same budget. That is encouraging in light of ARC’s earlier “roughly match sampling” examples, but is not a strict win. The assumptions were developed using random tests, and the method fails when noise is added to the weights.

*Squared error is the scoring rule used here: square the gap between an estimated accuracy and the true accuracy, then average across networks. Lower is better. The first candidate was tested at a budget equivalent to roughly 4 million complete network runs; the second at about 3,300. These compare counted computation, not elapsed runtime.*

The [results ledger](INDEX.md) explains what has been checked, what the limitations mean, and where to find the detailed evidence. A solution meeting the strictest reading of the full challenge remains open.

<details>
<summary>Repository guide and reproduction instructions</summary>

## Start here

The same table is in [AGENTS.md "Where things live"](AGENTS.md#where-things-live); each kind of information has one home.
New agents: read [AGENTS.md](AGENTS.md) (rules, standing owner rules), then [TASKS.md](TASKS.md), then [INDEX.md](INDEX.md). ([CLAUDE.md](CLAUDE.md) only imports AGENTS.md for Claude Code.)

| Document | Its single purpose |
|---|---|
| [README.md](README.md) | Entry point: what the project is, where to start, folders, how to reproduce |
| [AGENTS.md](AGENTS.md) | Rules that bind every agent, including the owner's standing rules (section "Standing rules") |
| [TASKS.md](TASKS.md) | Task board: what is in progress, who holds it, the backlog. Claim here before starting work |
| [OWNER_QA.md](OWNER_QA.md) | The owner's questions, the answers, and every owner directive verbatim (the only place they are recorded) |
| [INDEX.md](INDEX.md) | Experiment registry and hypothesis ledger: the only place for result numbers and claim status |
| [STATUS.md](STATUS.md) | One page: the current state in plain words, linking to INDEX rows |
| [docs/PROTOCOLS.md](docs/PROTOCOLS.md) | How work is run and scored: continuous work, ladder protocol, DRI directives 1-8 (rung test, two bars) |
| [DRI_HANDBOOK.md](DRI_HANDBOOK.md) | DRI process and lessons learned; read first if you take over as DRI |
| [m16/FINDINGS.md](m16/FINDINGS.md) | Narrative: how M_16,10 computes the 2nd argmax, error anatomy, estimators, failed hypotheses |
| [RESULTS.md](RESULTS.md) | Write-up of the AlgZoo legibility study |
| [PREREGISTRATION.md](PREREGISTRATION.md) | Hypotheses and thresholds committed before data was collected |
| [docs/archive/](docs/archive/) | History: old status updates, finished and superseded board text, later-update notes from the Q&A |

## ID prefixes and names

- **T** (T402): a task on the board ([TASKS.md](TASKS.md)); its result is registered once in [INDEX.md](INDEX.md).
- **E** (E33): an experiment in the INDEX registry under the older numbering.
- **H-xN** (H-R3): a claim in the INDEX hypothesis ledger, with its status. The letter is the topic: R rare events,
  E estimators, A other architectures, M mechanism, S strict bar, L lenient bar, C certification tools.
- **Q** (Q4): a top-level research question (INDEX section 1). **L** numbers are reserved by the ID rule but unused.
- **E1, E2:** the two rare-mistake types (E1: answering with the 3rd-largest input while the 2nd and 3rd differ by at least t0; E2: the interior near-gap mistake).
- **K3, S14, Q16, M5t, T5/gl20 and similar:** names of method variants inside one task, chosen by its author (not board IDs).
- **DEV3, HP4, HP5:** panels of retrained models; DEV3 is open for development, HP panels are sealed and scored once.
- **F:** our error divided by plain sampling's error at the same compute; strict bar F <= 1, lenient bar F <= 10.

## Folders

| Folder | Contents |
|---|---|
| [m16/](m16/README.md) | Analysis of M_16,10: shared code, ARC-protocol harness, leaderboard |
| [m16/approaches/](m16/approaches/README.md) | Estimator approaches to ARC's challenge |
| [algzoo/](algzoo/README.md) | Pre-registered legibility test (E3-E5): investigator labels + held-out scorer |
| [lit/](lit/README.md) | Literature scans by era and by field |
| [results/](results/README.md) | Machine-readable results (leaderboards, truths, AlgZoo scores) |
| [review/](review/README.md) | Independent critical review (R1) scripts |
| [tools/](tools/README.md) | Doc tooling (link checker, linkifier) |

## Reproduce

```
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install transformers pyarrow numpy scipy
pip install -e /path/to/alg-zoo            # https://github.com/alignment-research-center/alg-zoo
python m16/leaderboard.py truths && python m16/leaderboard.py run sampling stratified control_variate
python tools/check_links.py                # verify doc links
```

</details>
