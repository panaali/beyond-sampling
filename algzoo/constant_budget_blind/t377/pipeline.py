"""T377 pipeline: collect+audit runs, score with the shared scorer, build blinded judge batches.

Usage: python pipeline.py <scratch_t377_dir> <t377_assign.json> {collect|score|judging}
"""
import itertools
import json
import re
import shutil
import string
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
src, assign, step = Path(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3]
SEED = {"M16_s0": "0", "M16_s1": "1", "M16_example": "example", "M32_s1": "1", "M32_s2": "2", "M32_s0_arm160": "0"}

if step == "collect":
    WEIGHT = re.compile(r"alg_zoo|torch|weight|state_dict|\.rnn|_model|\.pth")
    OUTSIDE = re.compile(r"/home/user/clean-neurons|alignment-research-center|\.\./")
    audit = {}
    for code, a in sorted(assign.items()):
        d, out = src / code, ROOT / "runs" / code
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        rec = dict(a, files=[], weight_hits={}, outside_path_hits={})
        for p in sorted(d.iterdir()):
            if p.name in ("probe.py", "__pycache__", "TASK.md") or p.is_dir() or p.stat().st_size > 2_000_000:
                continue
            shutil.copy(p, out / p.name)
            rec["files"].append(p.name)
            if p.suffix in (".py", ".sh", ".md", ".txt"):
                t = p.read_text(errors="replace")
                if p.suffix in (".py", ".sh") and WEIGHT.findall(t):
                    rec["weight_hits"][p.name] = sorted(set(WEIGHT.findall(t)))
                if OUTSIDE.findall(t):
                    rec["outside_path_hits"][p.name] = sorted(set(OUTSIDE.findall(t)))
        ns = {}
        exec(compile((out / "labels.py").read_text(), code, "exec"), ns)
        rec["n_labels"] = len(ns["LABELS"])
        rec["noncompliant_constants"] = [k for k, v in ns.get("CONSTANTS", {}).items() if v > 5]
        rec["self_categories"] = {int(k): v[0] for k, v in ns["LABELS"].items()}
        audit[code] = rec
    json.dump(audit, open(ROOT / "results" / "audit.json", "w"), indent=1)
    for c, r in audit.items():
        print(c, r["model"], r["n_labels"], r["noncompliant_constants"], "W:", r["weight_hits"], "O:", r["outside_path_hits"])

elif step == "score":
    for code, a in sorted(assign.items()):
        out = ROOT / "results" / f"score_{code}.json"
        r = subprocess.run([sys.executable, str(ROOT / "score_seed.py"), str(ROOT / "runs" / code / "labels.py"), str(out),
                            str(a["H"]), SEED[a["model"]]], capture_output=True, text=True)
        if r.returncode:
            print(code, "FAILED", r.stderr[-1500:]); continue
        s = json.load(open(out))
        print(code, a["model"], f"model={s['model_acc']:.4f} repl={s['replacement_acc']:.4f} gap={s['gap_recovered']:.4f} "
              f"r2>=.8 {sum(v >= .8 for v in s['r2'])}/{s['n_neurons']}", flush=True)

elif step == "judging":
    J = ROOT / "judging"
    (J / "prompts").mkdir(parents=True, exist_ok=True)
    (J / "raw").mkdir(exist_ok=True)
    rng = np.random.default_rng(3770)
    REDACT = [r"[Nn]eurons? ?#?\d+", r"\d+ neurons", r"h\[\d+\]", r"h_10\[\d+\]"]
    codes = sorted(c for c, a in assign.items() if a["budget"] == 80)  # arm runs are not judged
    labels, red = {}, []
    for c in codes:
        ns = {}
        exec(compile((ROOT / "runs" / c / "labels.py").read_text(), c, "exec"), ns)
        for i, (cat, text) in ns["LABELS"].items():
            t = " ".join(str(text).split())
            for pat in REDACT:
                red += [{"code": c, "neuron": int(i), "removed": m} for m in re.findall(pat, t)]
                t = re.sub(pat, "[redacted]", t)
            labels[(c, int(i))] = t
    used = set()
    def nid():
        while True:
            s = "".join(rng.choice(list(string.ascii_uppercase + string.digits), 5))
            if s not in used:
                used.add(s); return s
    role = [{"id": nid(), "code": c, "neuron": i, "text": t} for (c, i), t in labels.items()]
    pair = []
    for model in sorted({assign[c]["model"] for c in codes}):
        mc = sorted(c for c in codes if assign[c]["model"] == model)
        H = assign[mc[0]]["H"]
        for a, b in itertools.combinations(mc, 2):
            while True:
                perm = rng.permutation(H)
                if (perm != np.arange(H)).all():
                    break
            for i in range(H):
                for kind, j in (("matched", i), ("control", int(perm[i]))):
                    P, Q = (a, i), (b, j)
                    if rng.random() < 0.5:
                        P, Q = Q, P
                    pair.append({"id": nid(), "kind": kind, "model": model, "H": H, "P": list(P), "Q": list(Q),
                                 "pair": [a, b], "neuron": i})
    def batches(items, size=28):
        o = rng.permutation(len(items)); items = [items[k] for k in o]
        return [items[k:k + size] for k in range(0, len(items), size)]
    rb, pb = batches(role), batches(pair)
    PR = ROOT.parent / "prompts"
    for name, bs, tmpl, fmt in [
        ("role", rb, (PR / "judge_role.md").read_text(), lambda it: f"{it['id']}: {it['text']}"),
        ("pair", pb, (PR / "judge_pair.md").read_text(),
         lambda it: f"{it['id']}:\n  P: {labels[tuple(it['P'])]}\n  Q: {labels[tuple(it['Q'])]}"),
    ]:
        for k, b in enumerate(bs):
            for jj, seq in ((1, b), (2, b[::-1])):
                (J / "prompts" / f"{name}_{k:02d}_j{jj}.txt").write_text(tmpl.replace("__ITEMS__", "\n".join(fmt(it) for it in seq)))
    json.dump({"role": rb, "pair": pb}, open(J / "key.json", "w"), indent=1)
    json.dump(red, open(J / "redactions.json", "w"), indent=1)
    print(len(role), "role items /", len(rb), "batches;", len(pair), "pair items /", len(pb), "batches;", len(red), "redactions")
