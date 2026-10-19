"""T426: table of bnb runs (results/bnb_*.json) with the pre-registered projections. usage: python summarize.py"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def proj(r):
    T = r["T"]; st = r["steps"]; done = [s for s in st if s["complete"]]
    S = [s["survivors"] for s in done]
    if done and S[-1] == 0:
        return S, None, 0, 0.0, "fully pruned at k = %d" % done[-1]["k"]
    part = [s for s in st if not s["complete"]]
    if not done:
        return S, None, None, None, "k = 1 incomplete"
    K = done[-1]["k"]; prev = S[-2] if len(S) > 1 else 1
    g = S[-1] / prev
    if part and part[0]["parents_done"]:                           # censored step: scale up by parents done
        p = part[0]; est = p["survivors"] * p["parents"] / p["parents_done"]
        g = est / S[-1]; note = f"g from censored k = {p['k']} ({p['parents_done']}/{p['parents']} parents)"
    else:
        note = f"g from k = {K}"
    ST = S[-1] * g ** (T - K)
    last = done[-1]; spc = last["sec"] / max(last["children"], 1); cps = last["children"] / max(prev, 1)
    cpu = sum(spc * cps * S[-1] * g ** (j - K - 1) for j in range(K + 1, T + 1)) / 3600
    return S, g, ST, cpu, note


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "bnb_*.json"))):
        r = json.load(open(f)); S, g, ST, cpu, note = proj(r)
        feas = None if ST is None else (ST <= 1e5 and cpu <= 200)
        lost = [s["hits_lost"] for s in r["steps"] if s["hits_lost"] is not None]
        rows.append((r["tag"], r["H"], r["T"], tuple(r["tri"]), r["mc_hits"], "/".join(map(str, S)),
                     "-" if g is None else f"{g:.2f}", "-" if ST is None else f"{ST:.2g}",
                     "-" if cpu is None else f"{cpu:.2g}", feas, max(lost) if lost else "-", round(r.get("cpu_sec", 0) / 3600, 2), note))
    print("| model | H | T | triple | MC hits | survivors k = 1.. (complete) | g | S_T proj | CPU-h proj | feasible | hits lost | CPU-h used | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(v) for v in r) + " |")


if __name__ == "__main__":
    main()
