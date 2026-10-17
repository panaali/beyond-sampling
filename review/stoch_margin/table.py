"""Print the README tables from the saved JSON files. usage: python review/stoch_margin/table.py"""
import glob
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
f = lambda r: "%+.2f +- %.2f (abs %.2f)" % (r["bias_pp"], r["se_pp"], r["abs_pp"])
print("## fidelity")
for p in sorted(glob.glob(f"{D}/fid_*.json")):
    d = json.load(open(p))
    print(f"| {d['model']} | {d['K']} | {d['NS']} | {d['acc']:.4f} | {d['cmc_gain_ceiling']:.2f} | " +
          " | ".join(f(d[k]) for k in ["R0", "lin", "add", "gauss", "slin", "slin_r", "lin_r"]) + " |")
print("## leverage")
for p in sorted(glob.glob(f"{D}/lev_*.json")):
    d = json.load(open(p))
    ks = ["top1", "top2", "top3", "top4", "top5", "pos2", "pos3", "pos4", "pos3v12", "rec1", "rec2"]
    print(f"| {d['model']} | {d['acc']:.4f} | " + " | ".join("%.2f" % d[k]["cmc_gain_ceiling"] for k in ks) + " |")
    print("   leverage:", " ".join("%s %.3f+-%.3f" % (k, d[k]["leverage"], d[k].get("se_leverage", 0)) for k in ks))
print("## closure")
for p in sorted(glob.glob(f"{D}/clos_*.json")):
    d = json.load(open(p))
    ks = [f"once_{t}" for t in range(10)] + ["seqG", "seq2", "seq4", "seqfull"]
    print(f"| {d['model']} | " + " | ".join("%+.1f" % d[k]["bias_pp"] for k in ks) + " |")
print("## anova")
for p in sorted(glob.glob(f"{D}/anova_*.json")):
    d = json.load(open(p))
    print(f"| {d['model']} | {d['K']} | {d['NS']} | " +
          " | ".join(f(d[k]) for k in ["anova", "anova_extrap_inf", "anova_r", "lin_ref"]) + " |")
