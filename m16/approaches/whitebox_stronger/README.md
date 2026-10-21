# T475: stronger public White-Box estimators

## Question

Can a stronger, published estimator improve our first submission within the owner's deadline?

## Method

We reproduced pinned e30_a and logxio sources in separate supervised workers, sequentially and with one BLAS thread. `sources.json` records immutable upstream commits and SHA256 hashes; the worker checks each hash before import. `estimator_e30.py` and `estimator_logxio.py` are unchanged upstream files. Keep `LICENSE` with e30 derivatives and `LICENSE.logxio` with logxio.

The first evaluation used T474's existing public reference rows: mini 0/1 and full 0/1. These are development data for our comparison. Full rows 2/3 were frozen before fetching their references and used only for a final robustness check. Upstream authors may have trained on any of these public networks; none is a sealed evaluation.

Two alternative explanations were considered before the runs: published gains might depend on a different scorer version, and learned corrections might overfit public data. We tested current metered cost and timing, compared distinct public networks, and retained the training-overlap limitation.

## Status

Complete for the deadline sprint. Original e30 and logxio passed local full-shape checks. The e30 dtype-only derivative was bitwise identical to original e30 on mini 0, with identical metered FLOPs. Root coordinates packaging and official submissions.

The optional sixth Strassen level passed our local pilot, but independent review found upstream server timeout failures. It is retained as a documented experiment and is **not recommended for submission**.

## Results

The single numerical evidence files are:

- `results/scores.json`: original and derivative development scores, metered costs, timing, memory, and both required point comparisons against covariance.
- `results/fresh_scores.json`: frozen original e30 on full rows 2/3.
- `results/dtype_parity.json`: exact saved-array comparison for the dtype-only refinement.
- `results/l6_parity.json` and `results/dtype_l6_parity.json`: sixth-level pilot comparisons.
- `results/compute.json`: total worker CPU and peak memory.

Root registers the headline results in the project INDEX. No population confidence interval or official rank is inferred from this small panel.

From the repository root, using the T474 environment:

```sh
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/run.py --method e30 --width 1024 --seed 6319981554997072999 --output /tmp/e30_mini0.json
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/run.py --method logxio --width 1024 --seed 1778484118851160678 --output /tmp/logxio_full0.json
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/score.py
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/score_fresh.py
```

Use a new output path: the supervisor refuses overwrites. Seeds for each reference are stored in its JSON metadata. `score.py` reads committed prediction files; `score_fresh.py` separately scores the final robustness extension.

## Compute accounting

All estimator arithmetic uses the actual flopscope meter. Setup is separately metered. Workers retain CPU, wall time and RSS receipts. Initial workers had a 120 CPU-second / 150 wall-second cap and a 7.5 GB RSS guard after the owner authorized up to 8 GB per process for the sprint. Root coordinated the aggregate memory limit and other workers.

The initial experiment allowed eight CPU minutes. Root later authorized one baseline completion and a final bounded robustness check. The last model run finished before 23:56 UTC, ahead of the 23:57 stop rule. Downloads, supervisors and saved-array scoring are excluded from worker CPU totals.

## Claims

- [measured] Pinned original e30 and logxio run at full challenge shape within our local contract checks on the tested networks.
- [measured] The dtype-only e30 change preserves every saved prediction and FLOP count on its native mini-0 comparison.
- [inferred] Comparing dtype metadata as a string avoids a redundant float32 cast in the remote client, whose dtype metadata is a string. The independent reviewer checked that client behavior; native parity alone does not measure remote savings.
- [measured] The optional sixth Strassen level reduces local FLOPs on the pilot.
- [inferred] Its published server timeout history makes it unsuitable for this deadline submission despite that local gain.

## Alternatives not ruled out

Public training overlap, floating-point platform effects, differences between local and remote timing, scorer-version changes, and panel selection noise remain possible. The full rows 2/3 were untouched by our selection but remain public data. No covariance rerun was done on that final extension; covariance comparison bars refer to the earlier four-row panel. T476 records direct fresh-variant comparisons against e30.

## Failure modes

The residual-time limit, server wall-clock limit, per-process memory cap and numerical stability all matter. These estimators contain shape-specific paths and learned corrections. A native timing pass does not prove that a remote suite will pass.

The sixth-level follow-up was frozen before its local pilot: test mini 0, then expand only if finite, below the residual threshold and lower scoring. Before expansion, an independent reviewer found upstream WRITEUP E1 reporting substantial server failures. No further full-row L6 tests were run. The subsequent dtype-only derivative preserves the original five-level setting.

## Exhaustion checklist

Tried: original e30 on six public networks; original logxio on three; optional L6 on mini 0; the dtype metadata refinement; independent reviews of T476 candidate sources and saved predictions.

Not tried: original V29 after stronger descendants passed; full K4, which exceeds the challenge compute budget; parameter search, owned by the T476 worker; a wider L6 panel, rejected because of documented server failures; or a sealed reference campaign, outside the deadline scope. No estimator family is declared impossible or dead.

## Independent T476 reviews

`results/t476_independent_review.json` checks rank448 against pinned e30: only the provenance header and designated rank literal differ. It also records independent loading of all four complete block256 prediction matrices and bitwise equality with T474, including matching model hashes. The block256 source diff changes default environment settings into their literal defaults and enlarges the two free-output-row blocks and associated scratch allocations. Summed indices, ranks and coefficients are unchanged.

`results/t476_additional_source_review.json` checks rank384/192 and tau3.0: only the provenance header and listed literals differ from pinned e30. These are independent source and saved-output reviews, not independent estimator reruns.

## Final robustness extension

Before fetching, root froze public full rows 2/3, original e30, and the already-defined T476 rank448 and tau3.0 variants. The fetch had a 5 MB byte cap and a 30-second supervisor timeout; it read only projected reference columns. `references_fresh.py` and `results/fresh_references/full_provenance.json` preserve the pinned dataset revision and exact byte ranges. No estimator settings were adapted to the new results.
