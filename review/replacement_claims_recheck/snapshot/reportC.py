"""Print Phase C (v2) tables from results/phaseC2.json and results/phaseC_diag.json."""
import json
import numpy as np

R = json.load(open("results/phaseC2.json"))
F = R["families"]
short = [f.split(":")[0] for f in F]
E, V = np.array(R["eval_r2"]), np.array(R["val_r2"])
print("**Held-out (eval-set) R^2 of the validation-selected family, and the first family in list order reaching 0.99**\n")
print("| neuron | " + " | ".join(f"t={t}" for t in range(1, 11)) + " |")
print("|---" * 11 + "|")
for j in range(16):
    cells = []
    for t in range(1, 11):
        b = V[:, t, j].argmax()
        ok = np.where(E[:, t, j] >= 0.99)[0]
        cells.append(f"{E[b, t, j]:.4f} ({short[ok[0]] if len(ok) else '-'})")
    print(f"| n{j} | " + " | ".join(cells) + " |")
print("\n**t=10 held-out R^2 by family (eval set)**\n")
print("| family | " + " | ".join(f"n{j}" for j in range(16)) + " | median |")
print("|---" * 18 + "|")
for f in range(len(F)):
    print(f"| {short[f]} | " + " | ".join(f"{E[f, 10, j]:.3f}" for j in range(16)) + f" | {np.median(E[f, 10]):.4f} |")
for ch in ("best", "best_noH"):
    print(f"\n**Replacement ({ch}): agreement / accuracy** (base accuracy {R['base']['acc']:.4f})\n")
    print("single neurons: " + ", ".join(f"{k} {v['agree']:.3f}" for k, v in R["single"][ch].items()))
    print("\nsingle groups: " + ", ".join(f"{k} {v['agree']:.3f}" for k, v in R["group"][ch].items()) + "\n")
    print("| cumulative, added | agree | acc | agree x9max | x9second | x9below | blind |")
    print("|---|---|---|---|---|---|---|")
    for c in R["cumulative"][ch]:
        print(f"| +{c['added']} | {c['agree']:.4f} | {c['acc']:.4f} | {c['agree_x9max']:.3f} | {c['agree_x9second']:.3f} | "
              f"{c['agree_x9below']:.3f} | {c['agree_blind']:.3f} |")
print("\n**Cumulative by neuron (best, best single first)**: " +
      ", ".join(f"+{c['added']} {c['agree']:.3f}" for c in R["cumulative_by_neuron"]))
D = json.load(open("results/phaseC_diag.json"))
print("\n**Noise needed: i.i.d. Gaussian noise on the true pre_10, every neuron at R^2 = 1 - eps**\n")
print("| 1 - R^2 | agreement | accuracy |\n|---|---|---|")
for r in D["noise"]:
    print(f"| {r['1-R2']:g} | {r['agree']:.4f} | {r['acc']:.4f} |")
for k, v in D.items():
    if k != "noise":
        print(f"\nDirect logit fit, {k}: agreement {v['logit_fit_agree']:.4f}, accuracy {v['logit_fit_acc']:.4f}")
