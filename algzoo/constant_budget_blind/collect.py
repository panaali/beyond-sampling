"""Copy each investigator's deliverables into runs/<code>/ and audit them (blinding, weights, constants).

Usage: python algzoo/constant_budget_blind/collect.py <scratch_t24_dir> <assign.json>
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src, assign = Path(sys.argv[1]), json.load(open(sys.argv[2]))
WEIGHT = re.compile(r"alg_zoo|torch|weight|state_dict|\.rnn|_model|\.pth")
OUTSIDE = re.compile(r"/home/user/clean-neurons|alignment-research-center|\.\./")
audit = {}
for code, H in sorted(assign.items()):
    d, out = src / code, ROOT / "runs" / code
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec = {"H": H, "files": [], "weight_hits": {}, "outside_path_hits": {}}
    for p in sorted(d.iterdir()):
        if p.name in ("probe.py", "__pycache__") or p.is_dir():
            continue
        if p.stat().st_size > 2_000_000:
            rec.setdefault("skipped_large", []).append(p.name)
            continue
        shutil.copy(p, out / p.name)
        rec["files"].append(p.name)
        if p.suffix in (".py", ".sh", ".md", ".txt"):
            text = p.read_text(errors="replace")
            hits = sorted(set(WEIGHT.findall(text))) if p.suffix in (".py", ".sh") else []
            if hits:
                rec["weight_hits"][p.name] = hits
            out_hits = sorted(set(OUTSIDE.findall(text)))
            if out_hits:
                rec["outside_path_hits"][p.name] = out_hits
    lab = out / "labels.py"
    rec["has_labels"] = lab.exists()
    if lab.exists():
        ns = {}
        exec(compile(lab.read_text(), str(lab), "exec"), ns)
        rec["n_labels"] = len(ns["LABELS"])
        rec["constants"] = {int(k): v for k, v in ns.get("CONSTANTS", {}).items()}
        rec["noncompliant_constants"] = [k for k, v in rec["constants"].items() if v > 5]
        rec["self_categories"] = {int(k): v[0] for k, v in ns["LABELS"].items()}
    audit[code] = rec
json.dump(audit, open(ROOT / "results" / "audit.json", "w"), indent=1)
print(json.dumps(audit, indent=1))
