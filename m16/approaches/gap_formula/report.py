"""Markdown tables from results/phase1_V1_V2.json and phase1_V3 log values (V3 JSON lost to an OOM in its LOO step)."""
import json
R = json.load(open("results/phase1_V1_V2.json"))["results"]
for fam, rows in R.items():
    print(f"\n**{fam}** ({rows['target']['n_params']} parameters incl. 2 blind rates)\n")
    print("| model | truth | own-fit bias | CV var ratio | K=0 transfer bias | K=3 refit bias | LOO bias |")
    print("|---|---|---|---|---|---|---|")
    for m, v in rows.items():
        f = lambda k: f"{v[k]:+.5f}" if k in v else "–"
        print(f"| {m} | {v['truth']:.5f} | {v['own_fit_bias']:+.5f} | {v['var_ratio_cv_vs_sampling']:.3f} | "
              f"{f('transfer_K0_bias')} | {f('refit_K3_bias')} | {f('loo_bias')} |")
