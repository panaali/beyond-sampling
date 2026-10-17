"""T113: all tables of the README from out/*.json. usage: python report.py > out/report.log"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "out")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


cnt = load(os.path.join(O, "count.json")) or {}
mult = {int(k): v["mult"] for k, v in cnt.items()}

print("## Fresh noise seeds 80-83 (abs sigma 0.05, 4 x 64 copies per zoo seed)\n")
print("| rung | zoo | clean acc | clean err | noisy acc | RMSE | max | ops (exact) | MSE ratio [CI] | worst noise seed | x auto-count | regions | half-pieces | cells clean / mean | cells/regions | cells/half-pieces | resolved | rescale | truth diff | reimpl diff |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
trend = []
for H in (2, 3, 4, 5, 6, 8, 16):
    for fn in sorted(glob.glob(os.path.join(O, f"fresh_H{H}_abs0.05_*.json"))):
        d = load(fn)
        for k, s in sorted(d["zoo"].items()):
            c = s["clean"]; mu = mult.get(H)
            for q in s["rows"]:
                trend.append((H, int(k), q["cells"], q["regions"], q["half_pieces"], q["ops"], q["exact_ops"]))
            cr = np.mean([q["cells"] / q["regions"] for q in s["rows"]]); ch = np.mean([q["cells"] / q["half_pieces"] for q in s["rows"]])
            print(f"| ({H},3) | {k} | {c['truth']:.4f} | {100*(c['est']-c['truth']):+.4f} pp | {s['mean_truth']:.3f} | {100*s['rms']:.3f}% | "
                  f"{100*s['max_abs']:.2f}% | {s['ops']/1e3:.0f}k ({s['exact_ops']/1e3:.0f}k) | {s['ratio']:.4f} [{s['ci'][0]:.4f}, {s['ci'][1]:.4f}] | "
                  f"{max(s['per_seed']):.4f} | {'' if mu is None else f'{s['ratio']*mu:.4f}'} | {c['regions']} | {c['half_pieces']} | {c['cells']} / {s['cells']:.0f} | "
                  f"{cr:.2f} | {ch:.2f} | {s['resolved_frac']:.0%} | {s['max_rescale_change']:.0e} | {s['max_truth_diff']:.0e} | {s['max_reimpl_diff']:.0e} |")

if trend:
    T = np.array(trend, float)
    print("\n## Cells vs regions along H (all models of the fresh runs; mean over models)\n")
    print("| H | models | regions | half-pieces | recipe cells | cells/regions | cells/half-pieces | recipe ops / exact ops |")
    print("|---|---|---|---|---|---|---|---|")
    for H in sorted(set(T[:, 0])):
        R = T[T[:, 0] == H]
        print(f"| {int(H)} | {len(R)} | {R[:,3].mean():.0f} | {R[:,4].mean():.0f} | {R[:,2].mean():.0f} | {np.mean(R[:,2]/R[:,3]):.2f} | "
              f"{np.mean(R[:,2]/R[:,4]):.2f} | {np.mean(R[:,5]/R[:,6]):.2f} |")
    # log-log slope of cells/regions and of cells/half-pieces vs H, cluster bootstrap over (H, zoo seed) groups
    for lab, col in (("cells/regions", 3), ("cells/half-pieces", 4)):
        for Hs in ((4, 5, 6, 8), (4, 5, 6, 8, 16), (2, 3, 4, 5, 6, 8, 16)):
            M = T[np.isin(T[:, 0], Hs)]
            if len(set(M[:, 0])) < len(Hs):
                continue
            x = np.log(M[:, 0]); y = np.log(M[:, 2] / M[:, col])
            slope = np.polyfit(x, y, 1)[0]
            groups = [np.flatnonzero((M[:, 0] == H) & (M[:, 1] == z)) for H in Hs for z in range(5)]
            rng = np.random.default_rng(0); bs = []
            for _ in range(2000):
                ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                if len(set(M[ix, 0])) > 1:
                    bs.append(np.polyfit(x[ix], y[ix], 1)[0])
            # also per-zoo-seed-mean regression (5 points per H)
            print(f"- log({lab}) vs log H over H = {Hs}: slope {slope:+.2f} [95% cluster-bootstrap CI {np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")

print("\n## Other noise levels (fresh seeds, 32 copies per zoo seed; worst zoo seed in bold)\n")
print("| noise | rung | ratio zoo 0 / 1 / 2 / 3 / 4 | worst [CI] | max abs err |")
print("|---|---|---|---|---|")
for fn in sorted(glob.glob(os.path.join(O, "fresh_H*_*.json"))):
    d = load(fn)
    if d["mode"] == "abs" and d["sigma"] == 0.05:
        continue
    if len(d["zoo"]) < 5:
        continue
    zz = [d["zoo"][str(z)] for z in range(5)]; w = int(np.argmax([s["ratio"] for s in zz]))
    print(f"| {d['mode']} {d['sigma']} (seed {d['seeds'][0]}) | ({d['H']},3) | " + " / ".join(f"{s['ratio']:.4f}" for s in zz) +
          f" | {zz[w]['ratio']:.3f} [{zz[w]['ci'][0]:.3f}, {zz[w]['ci'][1]:.3f}] (zoo {w}) | {100*max(s['max_abs'] for s in zz):.2f} pp |")

print("\n## Knob sensitivity (rec112, auto-counted ops both sides; clean + 16 copies abs 0.05 seed 86)\n")
S = {H: load(os.path.join(O, f"sens_H{H}.json")) for H in (4, 5, 6, 8)}
if all(S.values()):
    print("| variant | " + " | ".join(f"({H},3) worst ratio (zoo) / max clean err / ops vs frozen" for H in S) + " |")
    print("|---|" + "---|" * len(S))
    for v in S[4]:
        cells = []
        for H, d in S.items():
            r = d[v]; w = max(r, key=lambda z: r[z]["ratio"])
            ce = max(abs(r[z]["clean_err"]) for z in r); oo = np.mean([r[z]["ops"] / d["frozen"][z]["ops"] for z in r])
            cells.append(f"{r[w]['ratio']:.3f} ({w}) / {100*ce:.3f} pp / x{oo:.2f}")
        print(f"| {v} | " + " | ".join(cells) + " |")

for part in ("split", "climb", "diag"):
    d = load(os.path.join(O, f"adv_{part}.json"))
    if not d:
        continue
    print(f"\n## Adversarial: {part}\n")
    if part == "split":
        print("| rung | n copies | worst err clean models | worst err noisy copies | orig worst |")
        print("|---|---|---|---|---|")
        for H in (4, 5, 6, 8):
            for n in (2, 4, 8):
                r = [q for q in d if q["H"] == H and q["n"] == n]
                if not r:
                    continue
                c = max((q for q in r if q["model"] == 0), key=lambda q: abs(q["worst_err"]))
                nz = max((q for q in r if q["model"] > 0), key=lambda q: abs(q["worst_err"]))
                o = max(r, key=lambda q: abs(q["err_orig"]))
                print(f"| ({H},3) | {n} | {100*c['worst_err']:+.3f} pp (zoo {c['zoo']}) | {100*nz['worst_err']:+.3f} pp (zoo {nz['zoo']}) | {100*o['err_orig']:+.3f} pp |")
    elif part == "climb":
        print("| rung | zoo | clean acc | attacked acc | error |")
        print("|---|---|---|---|---|")
        for q in d:
            print(f"| ({q['H']},3) | {q['zoo']} | {q['clean_acc']:.4f} | {q['acc']:.4f} | {100*q['err']:+.3f} pp |")
    else:
        print("| case | truth | RECIPE112 | dense test | joint test | dense+joint |")
        print("|---|---|---|---|---|---|")
        for q in d:
            print(f"| {q['case']} | {q['truth']:.4f} | " + " | ".join(f"{100*q[k]['err']:+.4f} pp ({q[k]['cells']})" for k in ("RECIPE112", "dense", "joint", "dense+joint")) + " |")
