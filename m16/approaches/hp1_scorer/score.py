"""T281 HP1 scorer (Directive 8 amendment + Directive 7 panel). Independent harness; reads the sealed truths.

usage: python score.py <name> <module_file> <func> <class_label> [budgets]
  <module_file> is a frozen submission file (or 'control' for the built-in plain-MC control); <func> is its entry
  point est(model, budget) -> (estimate, cost_flops), model = (Wi (16,), Wh (16,16), Wo (10,16)) float64.
Inputs (git-ignored, never committed): data/hp1/{weights.npz, manifest.json, truths.json}, hashes checked first.
Per-model raw rows go to data/hp1/raw_<name>.jsonl (local only: they would reveal the truths). Only group
aggregates are printed and written to results/<name>.json.
"""
import hashlib, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
D = os.path.join(ROOT, "data", "hp1")
FPE = 5948.0                                   # FLOPs per plain-sampling forward at (16,10), T196 convention
BUDGETS = (16_000, 64_000, 256_000)
CURVE = (1_000, 4_000, 16_000, 64_000, 256_000)          # Directive 8 amendment 2
SEED_IDX = {16_000: 0, 64_000: 1, 256_000: 2, 1_000: 3, 4_000: 4}   # control seed slot per budget (stable)
GROUPS = ("clean", "abs0.05", "rel0.005", "rel0.01")
NBOOT, BOOT_SEED = 10_000, 281
SHA = dict(weights="9bad715dad33f2e403675ebd552dc5512a14e9a8019a636575e7a5a0735fbd4c",
           manifest="fe0974012be395454e60b182e3552c36d8ec24940c2ddaa1ebb620302988ca64",
           truths="b610b9a29eee87bda0891f52ea6016b40cb7948a2b0360787895cf04f4c8a509")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_panel():
    z = np.load(os.path.join(D, "weights.npz"))
    hw = hashlib.sha256(z["Wi"].tobytes() + z["Wh"].tobytes() + z["Wo"].tobytes()).hexdigest()
    assert hw == SHA["weights"], "weights hash"        # manifest hashes the array bytes, not the npz file
    assert sha_file(os.path.join(D, "manifest.json")) == SHA["manifest"], "manifest hash"
    assert sha_file(os.path.join(D, "truths.json")) == SHA["truths"], "truths hash"
    man = json.load(open(os.path.join(D, "manifest.json")))["models"]
    tr = json.load(open(os.path.join(D, "truths.json")))["truths"]
    names = list(z["names"])
    assert names == [m["name"] for m in man]
    models = [(z["Wi"][k].copy(), z["Wh"][k].copy(), z["Wo"][k].copy()) for k in range(len(names))]
    return names, man, models, tr


def _argmax_mc(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def control_factory():
    """Plain MC: n = B iid N(0,1)^10 inputs, fixed seed per model (set by the harness), cost B * 5948 FLOPs."""
    state = {}

    def est(model, budget):
        rng = np.random.default_rng(state["seed"])
        ok, n = 0, int(budget)
        for i in range(0, n, 1 << 16):
            x = rng.standard_normal((min(1 << 16, n - i), 10))
            ok += int((_argmax_mc(model, x) == np.argsort(x, 1)[:, -2]).sum())
        return ok / n, n * FPE
    return est, state


def load_sub(path, func):
    spec = importlib.util.spec_from_file_location("hp1_sub_" + func, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return getattr(mod, func)


def run(name, path, func, label, budgets=BUDGETS):
    names, man, models, tr = load_panel()
    raw_p = os.path.join(D, f"raw_{name}.jsonl")
    done = set()
    if os.path.exists(raw_p):
        done = {(r["k"], r["B"]) for r in map(json.loads, open(raw_p))}
    if path == "control":
        est, state = control_factory()
    else:
        est, state = load_sub(path, func), None
    with open(raw_p, "a") as f:
        for B in budgets:
            t0 = time.time()
            for k, nm in enumerate(names):
                if (k, B) in done:
                    continue
                if state is not None:
                    state["seed"] = 2810000 + 1000 * SEED_IDX[B] + k
                t = time.process_time()
                e, c = est(models[k], B)
                f.write(json.dumps(dict(k=k, name=nm, B=B, est=float(e), cost=float(c),
                                        cpu=time.process_time() - t)) + "\n")
                f.flush()
            print(f"{name} B={B}: {time.time() - t0:.0f}s", flush=True)
    return aggregate(name, label, path, func, budgets)


def aggregate(name, label, path, func, budgets=BUDGETS):
    names, man, models, tr = load_panel()
    rows = {(r["k"], r["B"]): r for r in map(json.loads, open(os.path.join(D, f"raw_{name}.jsonl")))}
    kind = [m["kind"] for m in man]
    base = [m["base"] for m in man]
    rng = np.random.default_rng(BOOT_SEED)
    out = dict(name=name, label=label, file=path, func=func, budgets={})
    for B in budgets:
        res = {}
        for g in GROUPS:
            ks = [k for k in range(len(names)) if kind[k] == g]
            p = np.array([tr[names[k]]["acc"] for k in ks]); se = np.array([tr[names[k]]["se"] for k in ks])
            est = np.array([rows[k, B]["est"] for k in ks]); cost = np.array([rows[k, B]["cost"] for k in ks])
            C = cost / FPE
            e = (est - p) ** 2
            s0 = p * (1 - p) / C
            s = s0 + se ** 2                        # truth MC noise enters e on average as SE^2
            bases = sorted(set(base[k] for k in ks))
            cl = np.array([bases.index(base[k]) for k in ks])
            E = np.bincount(cl, e, len(bases)); S = np.bincount(cl, s, len(bases))
            idx = rng.integers(0, len(bases), (NBOOT, len(bases)))
            Rb = E[idx].sum(1) / S[idx].sum(1)
            res[g] = dict(n=len(ks), R=float(e.sum() / s.sum()), R_noSE=float(e.sum() / s0.sum()),
                          lo=float(np.percentile(Rb, 2.5)), hi=float(np.percentile(Rb, 97.5)),
                          bias_pp=float(100 * (est - p).mean()), rms_pp=float(100 * np.sqrt(e.mean())),
                          se2_share=float((se ** 2).sum() / s.sum()),
                          cost_ok=bool((cost >= 0.5 * B * FPE).all()), cost_min_fpe=float(C.min()),
                          cost_max_fpe=float(C.max()))
        valid = all(r["cost_ok"] for r in res.values())
        out["budgets"][B] = dict(groups=res, valid=valid,
                                 lenient=valid and all(r["hi"] <= 10 for r in res.values()),
                                 strict=valid and all(r["R"] <= 1 for r in res.values()),        # restated D8
                                 strict_sig=valid and all(r["hi"] <= 1 for r in res.values()))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"{name}.json"), "w"), indent=1)
    print(f"\n{name} [{label}]  R [95% CI]  bias pp  rms pp  cost FPE(min)  SE^2 share")
    for B, d in out["budgets"].items():
        print(f"B={B}: valid={d['valid']} lenient={d['lenient']} strict={d['strict']} strict_sig={d['strict_sig']}")
        for g, r in d["groups"].items():
            print(f"  {g:9s} {r['R']:8.3f} [{r['lo']:7.3f}, {r['hi']:7.3f}]  {r['bias_pp']:+.3f}  {r['rms_pp']:.3f}"
                  f"  {r['cost_min_fpe']:.0f} ok={r['cost_ok']}  {r['se2_share']:.4f}")
    return out


def aggregate_F(name, label, budgets=CURVE):
    """Amendment 2 figure of merit, per group: F = MSE * C_eff / mean p(1-p), C_eff = max(C, 1000 FPE) with C the
    group-mean charged cost; break-even n* = mean p(1-p) / MSE. 95% CI: cluster bootstrap over base models.
    F_se: same with the truth-MC term removed from the MSE (mean SE^2 subtracted)."""
    names, man, models, tr = load_panel()
    rows = {(r["k"], r["B"]): r for r in map(json.loads, open(os.path.join(D, f"raw_{name}.jsonl")))}
    kind = [m["kind"] for m in man]
    base = [m["base"] for m in man]
    rng = np.random.default_rng(BOOT_SEED)
    out = dict(name=name, label=label, rule="Directive 8 amendment 2", budgets={})
    for B in budgets:
        res = {}
        for g in GROUPS + ("all",):
            ks = [k for k in range(len(names)) if g == "all" or kind[k] == g]
            p = np.array([tr[names[k]]["acc"] for k in ks]); se = np.array([tr[names[k]]["se"] for k in ks])
            est = np.array([rows[k, B]["est"] for k in ks]); cost = np.array([rows[k, B]["cost"] for k in ks])
            C = cost.mean() / FPE
            Ce = max(C, 1000.0)
            e2, pq = (est - p) ** 2, p * (1 - p)
            mse = e2.mean()
            bases = sorted(set(base[k] for k in ks))
            cl = np.array([bases.index(base[k]) for k in ks])
            nb = len(bases)
            E, Q, N = np.bincount(cl, e2, nb), np.bincount(cl, pq, nb), np.bincount(cl, None, nb)
            idx = rng.integers(0, nb, (NBOOT, nb))
            Fb = E[idx].sum(1) / Q[idx].sum(1) * Ce          # mean/mean: the counts cancel
            res[g] = dict(n=len(ks), C=float(C), C_eff=float(Ce), F=float(mse * Ce / pq.mean()),
                          lo=float(np.percentile(Fb, 2.5)), hi=float(np.percentile(Fb, 97.5)),
                          F_se=float((mse - (se ** 2).mean()) * Ce / pq.mean()),
                          rms_pp=float(100 * np.sqrt(mse)), bias_pp=float(100 * (est - p).mean()),
                          n_star=float(pq.mean() / mse), cost_ok=bool((cost >= 0.5 * B * FPE).all()))
        grp = [res[g] for g in GROUPS]
        valid = all(r["cost_ok"] for r in grp)
        out["budgets"][B] = dict(groups=res, valid=valid, strict=valid and all(r["F"] <= 1 for r in grp),
                                 lenient=valid and all(r["F"] <= 10 for r in grp))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"{name}_F.json"), "w"), indent=1)
    print(f"\n{name} [{label}]  F [95% CI]  F_se  rms pp  bias pp  n*  C_eff")
    for B, d in out["budgets"].items():
        print(f"B={B}: valid={d['valid']} strict={d['strict']} lenient={d['lenient']}")
        for g, r in d["groups"].items():
            print(f"  {g:9s} {r['F']:8.3f} [{r['lo']:7.3f}, {r['hi']:7.3f}] {r['F_se']:8.3f}  {r['rms_pp']:.3f}"
                  f"  {r['bias_pp']:+.3f}  {r['n_star']:9.0f}  {r['C_eff']:.0f}")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    bud = tuple(int(float(b)) for b in a[4].split(",")) if len(a) > 4 else BUDGETS
    run(a[0], a[1], a[2], a[3], bud)
    aggregate_F(a[0], a[3])
