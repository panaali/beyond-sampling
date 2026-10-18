"""Error attribution for chosen settings against the sampled truth split (results/truth_parts.json, diagnostic).
rank_exact: bias by K class (blind K<=1, bulk 2..8, top 9..10) and the split structural / quadrature / subset
  (structural = est(Q=64, stride 1) - truth; quadrature = est(Q, stride 1) - est(Q=64, stride 1);
   subset = est(Q, stride) - est(Q, stride 1)), same kgroups.
observer, probe: component errors err23 / err12 / blind (+ unmodelled rank-4+) against the sampled components.
usage: python attrib.py -> results/attrib.json, tables"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402

RAW = [json.loads(l) for l in open(os.path.join(cf.HERE, "results", "sweep_raw.jsonl"))]
TP = json.load(open(os.path.join(cf.HERE, "results", "truth_parts.json")))
TR = cf.lb3_truths()
PK = np.array([__import__("math").comb(10, k) for k in range(11)]) / 1024


def rows(fam, **cfg):
    return {d["model"]: d for d in RAW if d["fam"] == fam and all(d["cfg"].get(k) == v for k, v in cfg.items())}


def grp_rms(err):            # err: {model: value}  -> {group: (mean pp, rms pp)}
    out = {}
    for g in cf.GROUPS:
        v = np.array([e for m, e in err.items() if cf.group_of(m) == g])
        out[g] = (round(100 * v.mean(), 2), round(100 * np.sqrt((v ** 2).mean()), 2))
    return out


def rx_attrib(Q, stride, kgroups):
    r = rows("rx", Q=Q, stride=stride, kgroups=kgroups); r1 = rows("rx", Q=Q, stride=1, kgroups=kgroups)
    r64 = rows("rx", Q=64, stride=1, kgroups=kgroups); res = {}
    parts = {"blind K<=1": [0, 1], "bulk K=2..8": list(range(2, 9)), "top K>=9": [9, 10]}
    for lab, ks in parts.items():
        res[lab] = grp_rms({m: sum(PK[k] * (r[m]["parts"][str(k)] - (1 - TP[m]["errK"][k] / PK[k])) for k in ks)
                            for m in r})
    res["structural (Q=64,s=1)"] = grp_rms({m: r64[m]["est"] - TR[m]["acc"] for m in r})
    res["quadrature (Q vs 64)"] = grp_rms({m: r1[m]["est"] - r64[m]["est"] for m in r})
    res["subset (stride vs 1)"] = grp_rms({m: r[m]["est"] - r1[m]["est"] for m in r})
    res["total"] = grp_rms({m: r[m]["est"] - TR[m]["acc"] for m in r})
    return res


def comp_attrib(fam, **cfg):
    r = rows(fam, **cfg); res = {}
    for m, d in r.items():
        tp = TP[m]; p = d["parts"]
        t23, t12, t4, tb = tp["pick_pos"][2], tp["pick_pos"][0], sum(tp["pick_pos"][3:]), sum(tp["pick_blind"])
        if fam == "obs":
            e = {"err23": p["e23"] - t23, "err12": p["e12"] - t12, "blind": p["blind"] - tb, "rank4+ (unmodelled)": -t4}
        else:
            s = 1 - 11 / 1024
            e = {"err23": s * p["23"] - t23, "err12": s * p.get("12", 0) - t12, "blind": p["blind"] - tb,
                 "rank4+ (unmodelled)": -t4}
        for k, v in e.items():                      # positive = too much error predicted = accuracy biased low
            res.setdefault(k, {})[m] = -v
    out = {k: grp_rms(v) for k, v in res.items()}
    out["total"] = grp_rms({m: d["est"] - TR[m]["acc"] for m, d in r.items()})
    return out


def show(title, res):
    print(f"\n{title}  (accuracy-error contribution, mean / RMS pp per group)")
    print(f"{'component':24s}" + "".join(f"{g:>15s}" for g in cf.GROUPS))
    for k, v in res.items():
        print(f"{k:24s}" + "".join(f"{v[g][0]:+7.2f}/{v[g][1]:5.2f}".rjust(15) for g in cf.GROUPS))


def main(specs):
    out = {}
    for fam, cfg in specs:
        key = fam + ":" + ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        out[key] = rx_attrib(**cfg) if fam == "rx" else comp_attrib(fam, **cfg)
        show(key, out[key])
    json.dump(out, open(os.path.join(cf.HERE, "results", "attrib.json"), "w"), indent=1)


if __name__ == "__main__":
    main([("rx", dict(Q=4, stride=1, kgroups=1)), ("rx", dict(Q=16, stride=1, kgroups=11)),
          ("rx", dict(Q=4, stride=16, kgroups=3)),
          ("obs", dict(q=2, nq=12, rbins=6, stride=1)), ("obs", dict(q=1, nq=6, rbins=3, stride=1)),
          ("obs", dict(q=1, nq=6, rbins=3, stride=24)),
          ("probe", dict(version=2, iters=12, stride=1))])
