# clean-neurons

A tiny recurrent neural network—16 neurons and 432 parameters—reads 10 numbers and identifies the second-largest. ARC's challenge is to predict its accuracy from its weights and mechanism, competitively with random sampling at equal compute.

**[Interactive Tutorial](https://claude.ai/artifact/22zmMZh9WU6LAPLWqJmXRX)**

Some of our most interesting results:

- **Mechanistic estimation works on a smaller model:** the M_4,3 estimator achieved about 50× lower mean-squared error than sampling under its tested operation-count protocol (H-U1; scoped to the studied model and perturbations).
- **Structure helps at the full challenge size:** exact line walks combined with fixed-node quadrature achieved an MSE ratio of 0.540 [0.343, 0.762] versus equal-compute sampling on a sealed panel of clean retrains at 4 million forward-pass equivalents (H-E17). This is not a complete deductive solution.
- **Rare failures can be targeted:** a 12-weight edit made one studied rare mistake 17–19× rarer, independently reproduced, but also displaced probability into another mistake type (H-R3).

The challenge remains open. See the [results ledger](INDEX.md#4-hypothesis-ledger) for evidence and qualifications, and [current status](STATUS.md) for ongoing work.

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
| [m16/approaches/](m16/approaches/README.md) | 18 estimator attempts at ARC's challenge, one folder per agent |
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
