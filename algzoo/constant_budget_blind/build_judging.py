"""Build blinded judge batches from runs/*/labels.py (frozen rules in README).

Writes judging/key.json (secret mapping, never shown to judges), judging/redactions.json and
judging/prompts/{role,pair}_<batch>_<judge>.txt (judge 2 sees the batch reversed).
"""
import itertools
import json
import re
import string
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
J = ROOT / "judging"
(J / "prompts").mkdir(parents=True, exist_ok=True)
audit = json.load(open(ROOT / "results" / "audit.json"))
rng = np.random.default_rng(2410)
REDACT = [r"[Nn]eurons? ?#?\d+", r"\d+ neurons", r"h\[\d+\]", r"h_10\[\d+\]"]

labels, redactions = {}, []
for code in sorted(audit):
    ns = {}
    src = (ROOT / "runs" / code / "labels.py").read_text()
    exec(compile(src, code, "exec"), ns)
    for i, (cat, text) in ns["LABELS"].items():
        t = " ".join(str(text).split())
        for pat in REDACT:
            for m in re.findall(pat, t):
                redactions.append({"code": code, "neuron": int(i), "removed": m})
            t = re.sub(pat, "[redacted]", t)
        labels[(code, int(i))] = t

used = set()
def new_id():
    while True:
        s = "".join(rng.choice(list(string.ascii_uppercase + string.digits), 5))
        if s not in used:
            used.add(s)
            return s

# role items
role = [{"id": new_id(), "code": c, "neuron": i, "text": t} for (c, i), t in labels.items()]
# pair items
pair = []
for H in (8, 16, 32):
    codes = sorted(c for c in audit if audit[c]["H"] == H)
    for a, b in itertools.combinations(codes, 2):
        while True:
            perm = rng.permutation(H)
            if (perm != np.arange(H)).all():
                break
        for i in range(H):
            for kind, j in (("matched", i), ("control", int(perm[i]))):
                P, Q = (a, i), (b, j)
                if rng.random() < 0.5:
                    P, Q = Q, P
                pair.append({"id": new_id(), "kind": kind, "H": H, "P": list(P), "Q": list(Q),
                             "pair": [a, b], "neuron": i})

def batches(items, size=28):
    order = rng.permutation(len(items))
    items = [items[k] for k in order]
    return [items[k:k + size] for k in range(0, len(items), size)]

role_b, pair_b = batches(role), batches(pair)
tmpl_r = (ROOT / "prompts" / "judge_role.md").read_text()
tmpl_p = (ROOT / "prompts" / "judge_pair.md").read_text()
for name, bs, tmpl, fmt in [
    ("role", role_b, tmpl_r, lambda it: f"{it['id']}: {it['text']}"),
    ("pair", pair_b, tmpl_p, lambda it: f"{it['id']}:\n  P: {labels[tuple(it['P'])]}\n  Q: {labels[tuple(it['Q'])]}"),
]:
    for k, b in enumerate(bs):
        for judge, seq in ((1, b), (2, b[::-1])):
            (J / "prompts" / f"{name}_{k:02d}_j{judge}.txt").write_text(
                tmpl.replace("__ITEMS__", "\n".join(fmt(it) for it in seq)))
json.dump({"role": role_b, "pair": pair_b}, open(J / "key.json", "w"), indent=1)
json.dump(redactions, open(J / "redactions.json", "w"), indent=1)
print(len(role), "role items in", len(role_b), "batches;", len(pair), "pair items in", len(pair_b),
      "batches;", len(redactions), "redactions")
