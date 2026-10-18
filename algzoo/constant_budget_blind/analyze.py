"""Apply the frozen T24 scoring rules to scores and raw judge outputs; writes results/summary.json."""
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
J = ROOT / "judging"
audit = json.load(open(ROOT / "results" / "audit.json"))
key = json.load(open(J / "key.json"))
score = {c: json.load(open(ROOT / "results" / f"score_{c}.json")) for c in audit}
ROLES = {"MAX", "SECOND", "GAP", "POS_MAX", "POS_SECOND", "LAST", "RAW", "OTHER"}
PV = {"SAME": 1.0, "PARTIAL": 0.5, "DIFFERENT": 0.0}


def parse(kind):
    """{judge: {item_id: fields}} from judging/raw/<kind>_<batch>_j<judge>.txt"""
    out = {1: {}, 2: {}}
    for p in sorted((J / "raw").glob(f"{kind}_*_j*.txt")):
        judge = int(re.search(r"_j(\d)", p.name).group(1))
        for line in p.read_text().splitlines():
            parts = [s.strip().strip("*`") for s in line.split("|")]
            if len(parts) < 2 or not re.fullmatch(r"[A-Z0-9]{5}", parts[0]):
                continue
            out[judge][parts[0]] = [s.upper() for s in parts[1:]]
    return out


def kappa(a, b):
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / len(a) ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


role_raw, pair_raw = parse("role"), parse("pair")
role_items = [it for b in key["role"] for it in b]
pair_items = [it for b in key["pair"] for it in b]
rid = {(it["code"], it["neuron"]): it["id"] for it in role_items}
missing = {"role": {j: sum(it["id"] not in role_raw[j] for it in role_items) for j in (1, 2)},
           "pair": {j: sum(it["id"] not in pair_raw[j] for it in pair_items) for j in (1, 2)}}
invalid = {"role": sum(1 for j in (1, 2) for v in role_raw[j].values() if len(v) < 2 or v[1] not in ROLES
                       or v[0] not in "ABCD"),
           "pair": sum(1 for j in (1, 2) for v in pair_raw[j].values() if v[0] not in PV)}


def judged_role(j, code, i):
    v = role_raw[j].get(rid[(code, i)])
    return (v[0], v[1]) if v and len(v) >= 2 and v[1] in ROLES and v[0] in "ABCD" else None


def pair_score(it):
    vals = [PV[pair_raw[j][it["id"]][0]] for j in (1, 2)
            if it["id"] in pair_raw[j] and pair_raw[j][it["id"]][0] in PV]
    return float(np.mean(vals)) if vals else None


rng = np.random.default_rng(2410)
summary = {"missing_judgments": missing, "invalid_judgments": invalid, "sizes": {}}
for H in (8, 16, 32):
    codes = sorted(c for c in audit if audit[c]["H"] == H)
    gaps = {c: score[c]["gap_recovered"] for c in codes}
    s = {"investigators": {}, "G": float(np.median(list(gaps.values())))}
    for c in codes:
        r2 = np.array(score[c]["r2"])
        legj = []
        for j in (1, 2):
            cats = [judged_role(j, c, i) for i in range(H)]
            legj.append(np.mean([(cat is not None and cat[0] in "AB") and r2[i] >= 0.8 for i, cat in enumerate(cats)]))
        selfc = audit[c]["self_categories"]
        s["investigators"][c] = {
            "replacement_acc": score[c]["replacement_acc"], "model_acc": score[c]["model_acc"],
            "gap_recovered": gaps[c], "passes_90": gaps[c] >= 0.90,
            "n_r2_ge_0.8": int((r2 >= 0.8).sum()), "median_r2": float(np.median(r2)),
            "legible_judge_frac": float(np.mean(legj)),
            "legible_self_frac": float(np.mean([selfc[str(i)] in "AB" and r2[i] >= 0.8 for i in range(H)])),
            "self_categories": Counter(selfc.values()),
            "noncompliant_constants": audit[c].get("noncompliant_constants", []),
        }
    # pair-judge agreement, bootstrap over neurons
    per_neuron = defaultdict(lambda: {"matched": [], "control": []})
    for it in pair_items:
        if it["H"] == H:
            v = pair_score(it)
            if v is not None:
                per_neuron[it["neuron"]][it["kind"]].append(v)
    neurons = sorted(per_neuron)
    def stat(ns):
        m = [v for n in ns for v in per_neuron[n]["matched"]]
        k = [v for n in ns for v in per_neuron[n]["control"]]
        return np.mean(m), np.mean(k)
    Sm, Sc = stat(neurons)
    boot = []
    for _ in range(10_000):
        ns = rng.choice(neurons, len(neurons))
        a, b = stat(ns)
        boot.append(a - b)
    s.update({"S_matched": float(Sm), "S_control": float(Sc), "D": float(Sm - Sc),
              "D_CI95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "S_matched_by_neuron": {int(n): float(np.mean(per_neuron[n]["matched"])) for n in neurons}})
    # role agreement between investigators on the same neuron
    ks, same_m, same_c = [], [], []
    for j in (1, 2):
        for a, b in itertools.combinations(codes, 2):
            ra = [judged_role(j, a, i) for i in range(H)]
            rb = [judged_role(j, b, i) for i in range(H)]
            ok = [i for i in range(H) if ra[i] and rb[i]]
            ks.append(kappa([ra[i][1] for i in ok], [rb[i][1] for i in ok]))
            same_m += [ra[i][1] == rb[i][1] for i in ok]
    for it in pair_items:
        if it["H"] == H and it["kind"] == "control":
            for j in (1, 2):
                p, q = judged_role(j, *it["P"]), judged_role(j, *it["Q"])
                if p and q:
                    same_c.append(p[1] == q[1])
    s.update({"role_kappa_investigators": float(np.nanmean(ks)), "same_role_matched": float(np.mean(same_m)),
              "same_role_control": float(np.mean(same_c)),
              "legible_judge_frac_mean": float(np.mean([v["legible_judge_frac"] for v in s["investigators"].values()])),
              "legible_self_frac_mean": float(np.mean([v["legible_self_frac"] for v in s["investigators"].values()])),
              "role_distribution": Counter(judged_role(j, c, i)[1] for j in (1, 2) for c in codes for i in range(H)
                                           if judged_role(j, c, i))})
    summary["sizes"][H] = s

# judge reliability
rel = {}
both = [it for it in role_items if judged_role(1, it["code"], it["neuron"]) and judged_role(2, it["code"], it["neuron"])]
rel["role_kappa_judges"] = kappa([judged_role(1, it["code"], it["neuron"])[1] for it in both],
                                 [judged_role(2, it["code"], it["neuron"])[1] for it in both])
rel["cat_kappa_judges"] = kappa([judged_role(1, it["code"], it["neuron"])[0] for it in both],
                                [judged_role(2, it["code"], it["neuron"])[0] for it in both])
pb = [it for it in pair_items if all(it["id"] in pair_raw[j] and pair_raw[j][it["id"]][0] in PV for j in (1, 2))]
rel["pair_exact_agreement_judges"] = float(np.mean([pair_raw[1][it["id"]][0] == pair_raw[2][it["id"]][0] for it in pb]))
rel["pair_kappa_judges"] = kappa([pair_raw[1][it["id"]][0] for it in pb], [pair_raw[2][it["id"]][0] for it in pb])
summary["judge_reliability"] = rel

# frozen verdicts
G8, G32 = summary["sizes"][8]["G"], summary["sizes"][32]["G"]
D8, D32 = summary["sizes"][8]["D"], summary["sizes"][32]["D"]
lo32 = summary["sizes"][32]["D_CI95"][0]
rec = "holds" if G32 >= 0.90 and G32 >= G8 - 0.05 else "degrades" if G32 < 0.90 and G32 < G8 - 0.05 else "mixed"
lab = "holds" if D32 >= D8 - 0.15 and lo32 > 0 else "degrades" if D32 < D8 - 0.15 else "mixed"
overall = ("holds up" if rec == lab == "holds" else
           "does not hold up" if "degrades" in (rec, lab) else "mixed/inconclusive")
summary["verdict"] = {"reconstruction": rec, "labels": lab, "overall": overall}
json.dump(summary, open(ROOT / "results" / "summary.json", "w"), indent=1, default=lambda o: dict(o))
print(json.dumps({H: {k: v for k, v in s.items() if k not in ("S_matched_by_neuron",)}
                  for H, s in summary["sizes"].items()}, indent=1, default=lambda o: dict(o)))
print(json.dumps({k: summary[k] for k in ("missing_judgments", "invalid_judgments", "judge_reliability", "verdict")},
                 indent=1))
