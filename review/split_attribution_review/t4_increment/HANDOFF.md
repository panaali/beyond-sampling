# Incremental T4 pruning review handoff (independent saved audit pending)

The sole saved-only audit passed: 15 new rows, ten partial groups, 15 already-reviewed T3 records reused once. Source review c2fc8c76/shared4cfef0d9; immutable gate6f202a5b/sharedeb52ecaf preceded raw e5a37806/shared77bdc670. No estimator, cap, mass, model, RNG or geometry was replayed.

| Method | Threshold | Seeds present | Mean retained cells | Max point ratio | Max reported bound-squared ratio | Matched T4/T3 cell ratio |
|---|---:|---|---:|---:|---:|---:|
| cap | 1e-08 | 0 | 6085 | 1.90372654e-06 | 1.14283195e-05 | 25.5672269 |
| cap | 1e-07 | 0 | 5921 | 1.97041326e-06 | 0.00733819 | 26.4330357 |
| cap | 1e-06 | 0,1 | 4563.5 | 0.0044990735 | 7.92504108 | 21.9927711 |
| cap | 1e-05 | 0,1 | 3909.5 | 0.00293162569 | 1167.92576 | 21.0754717 |
| cap | 0.0001 | 0,1 | 3147 | 0.0415886537 | 145059.673 | 18.4035088 |
| mass | 1e-08 | 0 | 5118 | 2.68005049e-06 | 0.0018979569 | 22.8482143 |
| mass | 1e-07 | 0 | 4107 | 1.24003531e-05 | 0.460119614 | 19.1915888 |
| mass | 1e-06 | 0 | 2851 | 0.0150361992 | 54.4499596 | 14.5459184 |
| mass | 1e-05 | 0,1 | 1469 | 1.2611418 | 3641.28526 | 9.21003135 |
| mass | 0.0001 | 0,1 | 676.5 | 108.028712 | 145314.651 | 5.66108787 |

Ratios use the saved point reference, the source FLOP counter and a continuous sampling allowance of FLOPs/682. They are not independent truth confidence bounds, exact integer-budget comparisons or complete native costs. Reported B is not upgraded to a universally outward or whole-estimator bound. The largest mass-pruning point ratio exceeds one at the two largest thresholds; the tighter finite settings retain substantial T4/T3 cell growth. No threshold or variant is selected for adoption.

Only seed0 has all ten settings; seed1 has three cap and two mass settings. Other seeds/settings remain missing. Matching T3 seeds are used for cell-growth comparisons. These finite growth ratios do not prove a scaling law at larger T. The T5 phase1 archive was not opened; no T5 pruning output exists at the frozen author pin.

Review CPU: 0.003998705 seconds. Recorded author sec sum 414.256762505 is wall time, not process CPU. The reviewed source, required dependency pins, raw files, and exclusive empty successful run log are retained. No failed numerical run occurred; the pre-run metadata membership weakness and repair57018bce are preserved in history.

Alternatives: a provably outward replacement cap, a more informative volume bound, different split scheduling and methods aggregating regions remain open. Fixed-pad proof gaps and incomplete floating recipe/mass accuracy cannot be repaired by a saved arithmetic check. DRI03:05 closes phase2 without certified-pruning adoption; this handoff does not classify an entire method family as impossible.
