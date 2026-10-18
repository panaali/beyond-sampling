| family | setting | label | C (FPE) | F_max | fused bias/RMS pp, F | torch bias/RMS pp, F |
|---|---|---|---|---|---|---|
| branch_bound | budget=1000 | I | 182 | 46.5 | +4.8/5.8, 46.5 | +4.0/5.3, 39.7 |
| cone_enum | budget=1000 | D | 1149 | 119.3 | +8.1/8.8, 119.3 | +7.9/8.4, 114.1 |
| probe_bisect | iters=8,stride=1,version=1 | D | 3931 | 76.5 | +3.3/3.8, 76.5 | +3.2/3.6, 72.5 |
| probe_bisect | iters=12,stride=8,version=1 | I | 628 | 21.6 | +3.3/4.0, 21.6 | +3.2/3.9, 21.5 |
| probe_v5a | budget=1000 | I | 1009 | 3.0 | +1.0/1.4, 2.6 | +1.1/1.5, 3.0 |
| probe_v5b | budget=1000 | I | 1018 | 4.3 | +0.3/1.8, 4.3 | +0.2/1.5, 3.2 |
| probe_v5q | budget=1000 | I | 1009 | 1.2 | +0.6/0.9, 1.0 | +0.5/0.9, 1.2 |
| propagation | budget=1000 | D | 9440 | 34801.9 | -43.2/44.5, 24891.9 | -48.4/50.8, 34801.9 |
| rank_exact | Q=8,kgroups=3,stride=1 | D | 25779 | 131.4 | -1.6/1.9, 131.4 | -1.2/1.5, 85.3 |
| rank_exact | Q=8,kgroups=3,stride=16 | I | 1790 | 16.9 | -1.3/2.5, 15.7 | -1.2/2.6, 16.9 |
| reliability | variant=L0_s4_m02_nosorm | I | 2652 | 17.3 | +1.2/1.9, 13.0 | +1.3/2.2, 17.3 |
| t274b_cone | m=2 | D | 1980 | 312.6 | -5.8/10.8, 312.6 | -6.4/10.6, 310.2 |

Pareto set in (C, F_max), all labels: branch_bound [budget=1000] C=182 F_max=46.5; probe_bisect [iters=4,stride=8,version=1] C=297 F_max=45.9; probe_bisect [iters=8,stride=8,version=1] C=462 F_max=22.9; probe_bisect [iters=12,stride=8,version=1] C=628 F_max=21.6; probe_v5q [budget=1000] C=1009 F_max=1.2

Pareto set in (C, F_max), all-deductive: cone_enum [budget=1000] C=1149 F_max=119.3; probe_bisect [iters=4,stride=1,version=1] C=2504 F_max=108.5; probe_bisect [iters=8,stride=1,version=1] C=3931 F_max=76.5

Per-group Pareto sets in (C, RMS), all labels:
- fused: branch_bound [budget=1000] C=182 RMS=5.85 F=46.5; probe_bisect [iters=4,stride=8,version=1] C=297 RMS=5.81 F=45.9; probe_bisect [iters=8,stride=8,version=1] C=462 RMS=4.10 F=22.9; probe_bisect [iters=12,stride=8,version=1] C=628 RMS=3.98 F=21.6; probe_v5q [budget=1000] C=1009 RMS=0.86 F=1.0; probe_v5q [budget=4000] C=4035 RMS=0.50 F=1.3
- torch: branch_bound [budget=1000] C=182 RMS=5.33 F=39.7; probe_bisect [iters=8,stride=8,version=1] C=462 RMS=4.04 F=22.8; probe_bisect [iters=12,stride=8,version=1] C=628 RMS=3.93 F=21.5; rank_exact [Q=8,kgroups=1,stride=16] C=695 RMS=3.92 F=21.5; probe_v5q [budget=1000] C=1009 RMS=0.91 F=1.2; probe_v5q [budget=4000] C=4035 RMS=0.39 F=0.9
