"""T277 runner (pre-registered in README). usage (repo root): python m16/approaches/rate_T/run.py truth|sci_tao
truth -> truths.json (4e7-sample MC per zoo(16,T,seed)); sci_tao -> rows.jsonl (resumable)."""
import importlib.util, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path[:0] = [os.path.join(ROOT, "m16"), os.path.join(ROOT, "m16", "approaches", "sci_tao")]
import numpy as np
from core import zoo
TS, SEEDS, BUD = (3, 4, 5, 6, 8, 10), (0, 1, 2), (4000, 16000, 64000, 256000, 1000000)
TF = os.path.join(HERE, "truths.json"); RF = os.path.join(HERE, "rows.jsonl")
TR = json.load(open(TF)) if os.path.exists(TF) else {}

if sys.argv[1] == "truth":
    for T in TS:
        for s in SEEDS:
            k = f"T{T}_s{s}"
            if k in TR:
                continue
            Wi, Wh, Wo = zoo(16, T, s); rng = np.random.default_rng(9000 + 100 * T + s); ok = 0; n = 0; t0 = time.process_time()
            for _ in range(80):
                x = rng.standard_normal((500_000, T)); h = np.zeros((len(x), 16))
                for t in range(T):
                    h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
                ok += int(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).sum()); n += len(x)
            p = ok / n; TR[k] = dict(acc=p, se=float(np.sqrt(p * (1 - p) / n)), n=n, cpu=time.process_time() - t0)
            json.dump(TR, open(TF, "w"), indent=1); print(k, TR[k], flush=True)
else:
    spec = importlib.util.spec_from_file_location("st", "m16/approaches/sci_tao/estimator.py"); ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
    done = {(r["T"], r["seed"], r["budget"], r["rep"]) for r in map(json.loads, open(RF))} if os.path.exists(RF) else set()
    for T in TS:
        for s in SEEDS:
            m = zoo(16, T, s); tr = dict(TR[f"T{T}_s{s}"])
            ex = os.path.join(ROOT, "m16", "approaches", "width16_probe", "res", f"H16_T{T}_zoo{s}_exact.json")
            if os.path.exists(ex):                    # T = 3, 4: exact enumeration (T194), truth SE 0
                tr = dict(acc=json.load(open(ex))["acc"], se=0.0)
            for b in BUD:
                for r in range(8):
                    if (T, s, b, r) in done:
                        continue
                    ST.RULE_SEED = 4000 + 10 * r; t0 = time.process_time(); e, c = ST.est(m, b)
                    row = dict(T=T, seed=s, budget=b, rep=r, est=e, err=e - tr["acc"], truth_se=tr["se"], cost=2.23 * c, cpu=time.process_time() - t0)
                    with open(RF, "a") as f:
                        f.write(json.dumps(row) + "\n")
            print(T, s, "done", flush=True)
