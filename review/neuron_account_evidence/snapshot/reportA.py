"""Print Phase A tables (markdown) from results/phaseA.json."""
import json
import numpy as np

R = json.load(open("results/phaseA.json"))
b = R["base"]


def mat(mode, key, scale=100):
    M = np.array(R["matrices"][mode][key])
    ref = b[key]
    print(f"\n**{mode}-ablation, drop in {key} (pp) when h_t[j] is replaced at one step t**\n")
    print("| neuron | " + " | ".join(f"t={t}" for t in range(1, 11)) + " |")
    print("|---" * 11 + "|")
    for j in range(16):
        print(f"| n{j} | " + " | ".join(f"{scale * (ref - v):.1f}" for v in M[j]) + " |")


for mode in ["zero", "mean", "resample"]:
    mat(mode, "acc")
mat("resample", "agree")

print("\n**Ablation of a neuron/group at all steps, and at the final state only (accuracy; cases)**\n")
cols = ["acc", "agree", "acc_x9max", "acc_x9second", "acc_x9below", "acc_blind"]
print("| ablated | mode | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 2) + "|")
print("| none | - | " + " | ".join(f"{b[c]:.3f}" for c in cols) + " |")
for g, d in R["groups"].items():
    for mode in ["zero", "mean", "resample", "resample_final_only", "mean_final_only"]:
        print(f"| {g} | {mode} | " + " | ".join(f"{d[mode][c]:.3f}" for c in cols) + " |")
