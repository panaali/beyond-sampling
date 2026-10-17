"""Summarise D3/D3b: gain over MC on the skeleton integral = (Var_S(q)/N) / MSE_rule.
Var_S(q) from D1 (noise-corrected). usage: python m16/approaches/skeleton_quadrature/gains.py MODEL"""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
m = sys.argv[1]
vq = json.load(open(f"{D}/d1_{m}.json"))["var_q"]
print(f"{m}: Var_S(q) = {vq:.4f}")
for tag in ("", "_R24"):
    if not os.path.exists(f"{D}/d3{tag}_{m}.json"):
        continue
    d = json.load(open(f"{D}/d3{tag}_{m}.json"))
    print(f"d3{tag}: R = {d['R']} randomisations")
    print("| rule | N | gain (var over randomisations) | gain (MSE vs truth) | mean - truth (pp) |")
    for k, r in d["rules"].items():
        mc = vq / r["N"]
        print(f"| {k} | {r['N']} | {mc / max(r['var_rand'], 1e-12):.2f} | {mc / max(r['mse_vs_truth'], 1e-12):.2f} | "
              f"{100 * (r['mean'] - d['truth']):+.3f} |")
if os.path.exists(f"{D}/d3b_{m}.json"):
    d = json.load(open(f"{D}/d3b_{m}.json"))
    print("| product rule | N | error (pp) | noise s.d. (pp) | gain = MC var / err^2 |")
    for k, r in d["rules"].items():
        mc = vq / r["N"]
        e2 = max((r["err_pp"] / 100) ** 2 - (r["sd_noise_pp"] / 100) ** 2, (r["sd_noise_pp"] / 100) ** 2)
        print(f"| {k} | {r['N']} | {r['err_pp']:+.3f} | {r['sd_noise_pp']:.3f} | {mc / e2:.1f} |")
