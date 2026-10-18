"""T271 runner (pre-registered in README): error vs compute for deterministic families. Appends rows to rows.jsonl.
usage (repo root): python m16/approaches/rate_alpha/rates.py sci_tao|probe|form|enum"""
import importlib.util, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path[:0] = [os.path.join(ROOT, "m16")]
import numpy as np
import leaderboard as lb
KEY = 0.9531174789                                    # T192 3e11-sample answer key for the target
MODELS = ["orig", "s0.005_0", "s0.005_1", "s0.01_0", "s0.01_1"]
TR = json.load(open("results/leaderboard_truths.json")); ALL = lb.models()
truth = lambda n: KEY if n == "orig" else TR[n]["acc"]
OUT = os.path.join(HERE, "rows.jsonl")


def emit(**r):
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(r, flush=True)


def load(path, name):
    sys.path.insert(0, os.path.dirname(path))
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


fam = sys.argv[1]
DONE = {(r["fam"], r["model"], r["knob"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()   # resume
if fam == "sci_tao":
    ST = load("m16/approaches/sci_tao/estimator.py", "st")
    for n in MODELS:
        for b in (4000, 16000, 64000, 256000, 1000000):
            for s in range(8):
                ST.RULE_SEED = 4000 + 10 * s; t0 = time.process_time(); e, c = ST.est(ALL[n], b)
                emit(fam=fam, model=n, knob=b, rep=s, est=e, err=e - truth(n), cost=2.23 * c, cpu=time.process_time() - t0)
elif fam == "probe":
    import common, estimate as E
    for n in MODELS:
        if ("probe_v4", n, 16) in DONE:
            continue
        Wi, Wh, Wo = ALL[n]
        E.run = lambda x, Wi=Wi, Wh=Wh: common.run(x, Wi, Wh); E.Wo = Wo
        for s in (1, 2, 4, 8, 16):                    # v2: its own arrangement averaging
            if ("probe_v2", n, s) in DONE:
                continue
            t0 = time.process_time(); a, ev, _ = E.estimate_accuracy_v2(seeds=tuple(range(s)))
            emit(fam="probe_v2", model=n, knob=s, rep=0, est=a, err=a - truth(n), cost=float(ev), cpu=time.process_time() - t0)
        per = []
        for sd in range(16):                          # v4: one run per arrangement seed, averaged
            t0 = time.process_time(); a, ev, _ = E.estimate_accuracy_v4(seed=sd); per.append((a, ev, time.process_time() - t0))
        for s in (1, 2, 4, 8, 16):
            a = float(np.mean([p[0] for p in per[:s]])); ev = float(sum(p[1] for p in per[:s]))
            emit(fam="probe_v4", model=n, knob=s, rep=0, est=a, err=a - truth(n), cost=ev, cpu=float(sum(p[2] for p in per[:s])))
elif fam == "form":
    R = load("m16/approaches/reliability/estimator.py", "rel")
    for n in MODELS:
        for na in (1, 2, 4, 8, 16):
            if ("form", n, na) in DONE:
                continue
            t0 = time.process_time(); res = R.estimate(ALL[n], dict(R.CFG, n_arr=na)); e, c = res[0], res[1]
            emit(fam=fam, model=n, knob=na, rep=0, est=e, err=e - truth(n), cost=float(c), cpu=time.process_time() - t0)
elif fam == "enum":
    W = os.path.join(ROOT, "m16", "approaches", "width16_probe", "anatomy")
    for H in (8, 16):
        z = np.load(os.path.join(W, f"H{H}_T4_zoo0_chamber.npz")); J = json.load(open(os.path.join(W, f"H{H}_T4_zoo0_chamber.json")))
        geo = J["geometry_ops"] / J["leaves"]; o = np.argsort(z["leaf_mass"])[::-1]
        err, mo = z["err"][o], z["mass_ops"][o]; L = len(err); tot = err.sum()
        for frac in (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2):
            N = max(1, int(L * frac)); bias = float(tot - err[:N].sum())          # unkept error counted as correct
            emit(fam="enum_T4", model=f"zoo{H}_4", knob=N, rep=0, est=None, err=bias, cost=float(mo[:N].sum() + geo * N), cpu=0.0)
