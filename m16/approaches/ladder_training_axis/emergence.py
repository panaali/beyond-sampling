"""T101: when does the step-2 'x1 vs x0' switch appear? From mech_k*.json (clean models). For each seed and chart
(sign x0) print the switch ratios r in (0, 3) (switches at x1/x0 between 0 and 3), marking the one nearest 1.
usage: python emergence.py"""
import json
for s in range(5):
    print(f"seed {s}")
    for k in range(20, 31):
        M = json.load(open(f"mech_k{k}.json"))[str(s)]
        row = []
        for c in ("1", "-1"):
            r = sorted(x for x in M["charts"][c]["r"] if 0 < x < 3)
            row.append(f"x0{'>' if c == '1' else '<'}0 store {M['charts'][c]['store']} r(0,3) {[round(x, 2) for x in r]} |r-1|min {M['charts'][c]['near1']:.2f}")
        print(f"  k={k} acc {M['acc']:.3f} |Wi|max {M['Wi_abs_max']:5.1f}  " + " ; ".join(row))
