"""Score every runs/<code>/labels.py with the unchanged shared scorer (algzoo/score.py, seed 20261016)."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
audit = json.load(open(ROOT / "results" / "audit.json"))
timing = {}
for code in sorted(audit):
    out = ROOT / "results" / f"score_{code}.json"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(REPO / "algzoo/score.py"), str(ROOT / "runs" / code / "labels.py"),
                        str(out), "--zoo", f"{audit[code]['H']},10"], capture_output=True, text=True, cwd=REPO)
    timing[code] = {"wall_seconds": time.time() - t0, "returncode": r.returncode}
    if r.returncode:
        print(code, "FAILED", r.stderr[-2000:])
        continue
    s = json.load(open(out))
    print(code, audit[code]["H"], f"repl={s['replacement_acc']:.4f} model={s['model_acc']:.4f} "
          f"gap={s['gap_recovered']:.4f} r2>=.8: {sum(v >= 0.8 for v in s['r2'])}/{s['n_neurons']}")
json.dump(timing, open(ROOT / "results" / "score_timing.json", "w"), indent=1)
