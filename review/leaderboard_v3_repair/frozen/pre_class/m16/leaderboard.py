"""DRI-owned leaderboard (v3): every estimator scored on identical models, truth and compute accounting.

Interface. An approach folder that wants to be ranked exposes `m16/approaches/<name>/estimator.py` with
    est(model, budget, rng) -> (estimate, cost)
where model = (Wi, Wh, Wo) (numpy float64; Wi (H,), Wh (H,H), Wo (T,H); T = Wo.shape[0] = sequence length).
Optional module attributes:
    SUPPORTS = [(16, 10)]      (H, T) pairs the estimator handles (default: only M_16,10)
    FIXED_COST = True          cost does not depend on budget (scored at one budget only)
    ADVICE = {...}             see rule 3 below
    PRECOMPUTE_FLOPS = 0       see rule 4 below
`DEDUCTIVE` is no longer read: the class is assigned by the DRI after review (CLASS below, T14).

COMPUTE RULE (T16). `budget` and `cost` are in forward-pass equivalents (FPE).
 1. 1 FPE = 6e3 FLOPs x flops(H,T)/flops(16,10), where flops(H,T) = T(2H^2+2H) + 2HT is one forward pass of
    one length-T sequence. For M_16,10, 1 FPE = 6e3 FLOPs.
 2. Everything that depends on the model counts, converted at FLOPs/FPE: forward passes (full or partial),
    tangent/gradient/slope passes, bisections, pilot runs, per-model tuning or allocation, gate-pattern or
    region enumeration, LP/QP solves, linear algebra on the weights. If in doubt, it counts.
 3. ADVICE: any constant, table or allocation fitted on the outputs of a leaderboard (test) model is advice.
    Advice must be declared as ADVICE = {"fitted_on": [model names], "what": "..."}; rows for the models it was
    fitted on are flagged "tuned" and excluded from the headline numbers. Tune on the DEV models instead
    (`python m16/leaderboard.py dev` lists them; they are never scored).
 4. Model-free precomputation reused across all models (Gaussian tables, order statistics, fixed node sets) is
    free but must be declared in PRECOMPUTE_FLOPS (reported as a column, not charged).
 5. Generating ONE standard-normal input per model evaluation and its label (sort) is not charged, for any
    method, sampling included. Extra candidate inputs (rejection, stratum search) are charged at 0.04 FPE each.
 7. Cost is charged AS IMPLEMENTED, not as implementable: a saving (e.g. incremental re-use of a shared prefix)
    counts only once the code does it. Where the DRI audit (review/estimator_audit/) measured a gap between the
    reported and the implemented work, OVERRIDES below rescales the reported cost at report time (raw runs keep
    the estimator's own number). Implement the saving to remove the factor.
 6. The measured CPU time is reported next to cost, as the ratio to plain sampling's time at the same cost.
    A time ratio far above 1 means cost is probably under-counted and triggers a DRI audit.

CLASSES (T14). The headline question is ARC's: does a *mechanistic* estimator beat sampling?
    mechanistic  deterministic; uses the weights through analysis (regions, propagated distributions,
                 asymptotics, optimisation, LPs), not by averaging model outputs over a generic node set
    quadrature   deterministic, but essentially an average of model outputs over a fixed model-independent
                 node set (QMC, lattices, fixed lines): replacing the nodes by random points gives Monte Carlo
    sampling     uses random model inputs
Estimators not yet reviewed are reported as "unreviewed". Anything classed mechanistic/quadrature must pass a
determinism check (two different rngs give the same estimate) or it is reported as sampling.

SCORING. MSE = mean (estimate - truth)^2 - truth_se^2 (unbiased for the MSE against the exact accuracy),
reported as the ratio to plain sampling's MSE at the same cost, mean over runs of p_m(1-p_m)/cost_run with p_m the
run's model accuracy (T22; below 1 = better), with a cluster-bootstrap 95% CI (models resampled, then runs within each model). Every single run is stored in data/leaderboard_v3_raw.jsonl (copied to results/ by each report), so tables are recomputed
from raw estimates when truths improve (`report`), without re-running estimators.

usage:
  python m16/leaderboard.py truths [--H 16 --T 10] [--n 4000000]   # cache ground truth for one (H,T)
  python m16/leaderboard.py run NAME... [--H --T] [--budgets 16000,64000] [--groups orig,seed]
  python m16/leaderboard.py report [--H --T]                       # tables from stored raw runs
  python m16/leaderboard.py dev [--H --T]                          # list the DEV (tuning) models
NAME is an approach folder or one of: sampling, stratified.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # single-threaded: fair CPU-time column, light on the shared cores
import numpy as np  # noqa: E402

sys.path.insert(0, "m16")

VERSION = 3
REPS = 24
DEFAULT_BUDGETS = (16_000, 64_000)
CURVE_BUDGETS = (1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000)  # T15
SIGMAS = (0.0025, 0.005, 0.01, 0.02)                                  # x RMS of each matrix; T17 adds 0.01, 0.02
N_NOISY = 6
RAW = "data/leaderboard_v3_raw.jsonl"            # live, git-ignored: git checkouts/stashes must never touch an open file
RAW_SNAPSHOT = "results/leaderboard_v3_raw.jsonl"  # committed copy, refreshed by every report

# DRI-reviewed classes (T14). Reason and audit pointer for each; see review/estimator_audit/.
# Classes from the independent audit (review/estimator_audit/README.md, 2026-10-17); determinism verified there.
CLASS = {
    "sampling": ("sampling", "plain Monte Carlo baseline"),
    "stratified": ("sampling", "stratified Monte Carlo"),
    "control_variate": ("sampling", "srb: random prefixes + exact last step"),
    "tie_lines": ("sampling", "joint: random gap strata + antithetics"),
    "boundary": ("sampling", "random surface points and remainders"),
    "debiased_series": ("sampling", "stratified sampling of a residual"),
    "great_circle": ("sampling", "random great circles"),
    "rank_exact": ("mechanistic", "frozen gates per rank class, exact integration"),
    "propagation": ("mechanistic", "propagated Gaussian mixtures / cumulants"),
    "cone_enum": ("mechanistic", "error-cone enumeration + model-free cone masses"),
    "reliability": ("mechanistic", "FORM/SORM design points per chamber"),
    "observer": ("mechanistic", "signal-detection observer at canonical points"),
    "tube": ("quadrature", "fixed tie-slicing node set"),
    "sci_laplace": ("quadrature", "fixed Sobol node set on tie bands"),
    "sci_tao": ("quadrature", "fixed scrambled-Sobol faces"),
    "sci_feynman": ("quadrature", "fixed Sobol segment walks"),
    "branch_bound": ("quadrature", "below ~144k FPE only evaluates 180 fixed cell centres"),
}

# DRI rulings on the audit (rules 3, 5, 7). cost_factor: implemented / reported work (measured where stated).
# cost_add(budget): FPE added per call. advice: models the estimator's constants were fitted or selected on.
_TUNED_NOISY = [f"s{sg}_{k}" for sg in (0.0025, 0.005) for k in range(N_NOISY)]
OVERRIDES = {
    "tube": dict(cost_factor=2.06, advice=["orig"]),                 # measured (fullpass_count.out)
    "sci_laplace": dict(cost_factor=1.81, advice=["orig"]),          # measured
    "sci_tao": dict(cost_factor=2.23),                               # measured; no advice
    "sci_feynman": dict(cost_factor=2.0, advice=["orig"]),           # same scheme, not measured; SHARES provenance unknown
    "boundary": dict(cost_factor=2.0, advice=["orig"]),              # same scheme, not measured; GA/GB tuned on orig
    "control_variate": dict(cost_factor=3.3 / 1.6, advice=["orig"]),  # hand count rb9.py:158-176 vs :184
    "great_circle": dict(cost_factor=7 / 3, advice=["orig"]),        # hand count
    "tie_lines": dict(advice=["orig"]),
    "stratified": dict(advice=["orig"]),          # rejection now charged inside stratified.estimate (T21)
    "debiased_series": dict(advice=["orig"]),     # uses stratified.estimate below 128k (charged since T21)
    "observer": dict(fixed_cost=True, advice=["orig"] + _TUNED_NOISY),     # design chosen against these truths
    "reliability": dict(advice=["orig"] + _TUNED_NOISY),                    # ladder swept against these truths
    "propagation": dict(advice=["orig"]),                                   # budget->eps map only
}


# ------------------------------------------------------------------ models and truth
def flops(H, T):
    return T * (2 * H * H + 2 * H) + 2 * H * T


def fpe_flops(H, T):
    """FLOPs in one forward-pass equivalent for M_H,T (rule 1)."""
    return 6e3 * flops(H, T) / flops(16, 10)


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def correct(model, x):
    return forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]


def est_sampling(model, budget, rng):
    x = rng.standard_normal((int(budget), model[2].shape[0]))
    return float(correct(model, x).mean()), int(budget)


def _zoo(H, T, seed):
    from alg_zoo import zoo_2nd_argmax
    m = zoo_2nd_argmax(H, T, seed=seed)
    f = lambda t: t.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


def _noisy(base, k, sigma, seed):
    rng = np.random.default_rng(seed)
    return [tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape) for W in base)
            for _ in range(k)]


def _base(H, T):
    if (H, T) == (16, 10):
        from arc_eval import Wi0, Wh0, Wo0       # example_2nd_argmax (differs from training seeds 0-4)
        return "orig", (Wi0, Wh0, Wo0)
    from alg_zoo.zoo import best_of_5_seed_2nd_argmax
    s = best_of_5_seed_2nd_argmax(hidden_size=H, seq_len=T)
    return f"seed{s}", _zoo(H, T, s)


def models(H=16, T=10, dev=False):
    """Test models (scored) or DEV models (for tuning only: noisy copies drawn with a different seed)."""
    bname, base = _base(H, T)
    ms = {}
    if not dev:
        if bname == "orig":
            ms["orig"] = base
        for sd in range(5):                      # ARC: "a good method should also work well on" these seeds
            ms[f"seed{sd}"] = _zoo(H, T, sd)
    for sigma in SIGMAS:
        for k, m in enumerate(_noisy(base, N_NOISY, sigma, seed=1 if dev else 0)):
            ms[f"{'dev_' if dev else ''}s{sigma}_{k}"] = m
    return ms


def group_of(name):
    return "orig" if name == "orig" else "seed" if name.startswith("seed") else name.rsplit("_", 1)[0]


def truths_path(H, T):
    return "results/leaderboard_truths.json" if (H, T) == (16, 10) else f"results/leaderboard_truths_H{H}_T{T}.json"


def compute_truths(H, T, n=4_000_000, chunk=500_000):
    path = truths_path(H, T)
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name, m in models(H, T).items():
        if name in out and out[name]["n"] >= n:
            continue
        rng = np.random.default_rng(987_654 + zlib.crc32(f"{H},{T},{name}".encode()) % 10_000)
        ok = 0
        for _ in range(n // chunk):
            ok += int(correct(m, rng.standard_normal((chunk, T))).sum())
        p = ok / n
        out[name] = {"acc": p, "se": float(np.sqrt(p * (1 - p) / n)), "n": n}
        print(name, out[name], flush=True)
        json.dump(out, open(path, "w"), indent=1)


# ------------------------------------------------------------------ estimators
def load_estimator(name):
    """-> (est, attrs). attrs: supports, fixed_cost, advice, precompute_flops."""
    a = dict(supports=[(16, 10)], fixed_cost=False, advice=None, precompute_flops=0)
    if name == "sampling":
        a["supports"] = "all"
        return est_sampling, a
    if name == "stratified":
        import stratified as S
        d = json.load(open("m16/stratified_design.json"))
        P, alloc = np.array(d["P"]), np.array(d["alloc"])
        a["advice"] = {"fitted_on": OVERRIDES["stratified"]["advice"], "what": "stratum design (stratified_design.json)"}
        return (lambda m, b, rng: S.estimate(lambda x: forward(m, x), b, rng, P, alloc)), a
    path = f"m16/approaches/{name}/estimator.py"
    spec = importlib.util.spec_from_file_location(f"est_{name}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    a["supports"] = [tuple(s) for s in getattr(mod, "SUPPORTS", [(16, 10)])]
    a["fixed_cost"] = bool(getattr(mod, "FIXED_COST", False))
    a["advice"] = getattr(mod, "ADVICE", None)
    a["precompute_flops"] = float(getattr(mod, "PRECOMPUTE_FLOPS", 0))
    o = OVERRIDES.get(name, {})
    if "advice" in o:
        a["advice"] = {"fitted_on": o["advice"], "what": "DRI audit ruling"}
    a["fixed_cost"] = a["fixed_cost"] or o.get("fixed_cost", False)
    return mod.est, a


def _charged(name, cost, budget):
    o = OVERRIDES.get(name, {})
    return cost * o.get("cost_factor", 1.0) + (o["cost_add"](budget) if "cost_add" in o else 0.0)


def _seed(name, mname, r):
    return 10_000 * r + zlib.crc32(f"{name}|{mname}".encode()) % 1000


def run(names, H, T, budgets, groups):
    ms = models(H, T)
    if groups:
        ms = {k: v for k, v in ms.items() if group_of(k) in groups}
    os.makedirs("results", exist_ok=True); os.makedirs(os.path.dirname(RAW), exist_ok=True)
    for name in names:
        est, a = load_estimator(name)
        if a["supports"] != "all" and (H, T) not in a["supports"]:
            print(f"{name}: does not support (H,T)=({H},{T}); skipped"); continue
        cls = CLASS.get(name, ("unreviewed", ""))[0]
        det = cls in ("mechanistic", "quadrature")
        if det:     # determinism check on the first model, smallest budget
            m0 = next(iter(ms.values()))
            e1, _ = est(m0, budgets[0], np.random.default_rng(1)); e2, _ = est(m0, budgets[0], np.random.default_rng(2))
            det = abs(e1 - e2) < 1e-12
            if not det:
                print(f"{name}: FAILED determinism check ({e1} vs {e2}); scored as sampling", flush=True)
        with open(RAW, "a") as f:
            for b in budgets:
                for mname, m in ms.items():
                    for r in range(1 if det else REPS):
                        t0 = time.process_time()
                        e, c = est(m, b, np.random.default_rng(_seed(name, mname, r)))
                        f.write(json.dumps(dict(v=VERSION, name=name, H=H, T=T, model=mname, budget=b, rep=r,
                                                est=float(e), cost=float(c), cpu_s=time.process_time() - t0,
                                                det_ok=det)) + "\n")
                    f.flush()
                print(f"{name} budget {b} done", flush=True)
                if a["fixed_cost"]:
                    break
    report(H, T, names)


# ------------------------------------------------------------------ report
_SPF = {}


def _sampling_sec_per_fpe(H, T, budget):
    """CPU seconds per FPE of plain sampling run at this budget (same batch size, so overheads match)."""
    if (H, T, budget) not in _SPF:
        m = next(iter(models(H, T).values())); rng = np.random.default_rng(0)
        reps = max(1, 200_000 // budget); t0 = time.process_time()
        for _ in range(reps):
            est_sampling(m, budget, rng)
        _SPF[H, T, budget] = (time.process_time() - t0) / (reps * budget)
    return _SPF[H, T, budget]


def load_raw(H, T):
    """Latest run per (name, model, budget, rep): a re-run replaces earlier rows."""
    rows = {}
    if not os.path.exists(RAW) and os.path.exists(RAW_SNAPSHOT):     # fresh clone: start from the committed copy
        os.makedirs(os.path.dirname(RAW), exist_ok=True); shutil.copy(RAW_SNAPSHOT, RAW)
    if os.path.exists(RAW):
        for line in open(RAW):
            d = json.loads(line)
            if d["v"] == VERSION and d["H"] == H and d["T"] == T:
                rows[(d["name"], d["model"], d["budget"], d["rep"])] = d
    return list(rows.values())


def table(H, T, names=None):
    truth = json.load(open(truths_path(H, T)))
    raw = [d for d in load_raw(H, T) if names is None or d["name"] in names]
    out = []
    for name in sorted({d["name"] for d in raw}):
        _, a = load_estimator(name)
        tuned = set((a["advice"] or {}).get("fitted_on", []))
        cls = CLASS.get(name, ("unreviewed", ""))[0]
        rs = [d for d in raw if d["name"] == name]
        if cls in ("mechanistic", "quadrature") and not all(d["det_ok"] for d in rs):
            cls = "sampling"
        for b in sorted({d["budget"] for d in rs}):
            for g in sorted({group_of(d["model"]) for d in rs}, key=lambda g: (g != "orig", g)):
                sel = [d for d in rs if d["budget"] == b and group_of(d["model"]) == g]
                if not sel:
                    continue
                se2 = np.array([(d["est"] - truth[d["model"]]["acc"]) ** 2 - truth[d["model"]]["se"] ** 2 for d in sel])
                # T22: each run's own sampling MSE at its own charged cost, p_m(1-p_m)/C_run (not mean(p)(1-mean(p)))
                cst = np.array([_charged(name, d["cost"], b) for d in sel])
                v = np.array([truth[d["model"]]["acc"] * (1 - truth[d["model"]]["acc"]) for d in sel]) / cst
                C = float(cst.mean()); base = float(v.mean())
                # cluster bootstrap over models (runs on one model share its truth error), runs resampled within
                mods = sorted({d["model"] for d in sel}); by = {m: np.flatnonzero([d["model"] == m for d in sel]) for m in mods}
                brng = np.random.default_rng(0); bs = []
                for _ in range(2000):
                    ii = np.concatenate([brng.choice(by[m], len(by[m])) for m in brng.choice(mods, len(mods))])
                    bs.append(se2[ii].mean() / v[ii].mean())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out.append(dict(name=name, cls=cls, budget=b, group=g, cost=C, n=len(sel),
                                rmse_pp=100 * np.sqrt(max(se2.mean(), 0)), ratio=se2.mean() / base, ci=[lo, hi],
                                bias_pp=100 * np.mean([d["est"] - truth[d["model"]]["acc"] for d in sel]),
                                time_ratio=np.mean([d["cpu_s"] for d in sel]) / (C * _sampling_sec_per_fpe(H, T, b)),
                                tuned=bool(tuned & {d["model"] for d in sel}),
                                precompute=a["precompute_flops"]))
    return out


def report(H=16, T=10, names=None):
    if os.path.exists(RAW):
        shutil.copy(RAW, RAW_SNAPSHOT)
    rows = table(H, T, names)
    json.dump(rows, open(f"results/leaderboard_v3_H{H}_T{T}.json", "w"), indent=1)
    print(f"\n=== Leaderboard v{VERSION}, M_{H},{T}. MSE / sampling-MSE at equal cost (below 1 beats sampling) ===")
    print("    ARC's demonstrated bar (E32, M_4,3): MSE ratios ~2.4 (B) to ~10 (A) at ARC's op counts were called 'roughly match';"
          " A is ~1.4 when ops are counted as implemented (as here).")
    for cls in ("mechanistic", "quadrature", "sampling", "unreviewed"):
        sel = [r for r in rows if r["cls"] == cls]
        if not sel:
            continue
        print(f"\n--- {cls}{'  (ARC challenge: a row below 1 here)' if cls == 'mechanistic' else ''} ---")
        print(f"{'method':18s} {'group':9s} {'budget':>8s} {'cost':>9s} {'ratio':>6s} {'95% CI':>14s} "
              f"{'bias pp':>8s} {'time':>5s}  n")
        for r in sel:
            print(f"{r['name']:18s} {r['group']:9s} {r['budget']:8d} {r['cost']:9.0f} {r['ratio']:6.2f} "
                  f"[{r['ci'][0]:5.2f},{r['ci'][1]:5.2f}] {r['bias_pp']:+8.3f} {r['time_ratio']:5.1f} "
                  f"{r['n']:3d}{'  TUNED-ON (advice)' if r['tuned'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["truths", "run", "report", "dev"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--budgets", default=None, help="comma list, or 'curve' for the T15 grid")
    ap.add_argument("--groups", default=None, help="comma list of groups, e.g. orig,seed,s0.005")
    a = ap.parse_args()
    budgets = (CURVE_BUDGETS if a.budgets == "curve" else
               tuple(int(b) for b in a.budgets.split(",")) if a.budgets else DEFAULT_BUDGETS)
    groups = set(a.groups.split(",")) if a.groups else None
    if a.cmd == "truths":
        compute_truths(a.H, a.T, a.n)
    elif a.cmd == "run":
        run(a.names, a.H, a.T, budgets, groups)
    elif a.cmd == "report":
        report(a.H, a.T, a.names or None)
    else:
        print("\n".join(models(a.H, a.T, dev=True)))


if __name__ == "__main__":
    main()
