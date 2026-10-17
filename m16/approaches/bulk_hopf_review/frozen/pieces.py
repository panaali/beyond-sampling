"""Mean / max exact pieces per fibre at nu=16 (the 'cells' of the fibre method) per rung and seed -> pieces.json."""
import sys, os, json; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43")); from zoo44 import zoo; import hopf
out = {}
for H in (4, 5, 6):
    r = []
    for s in range(5):
        m = zoo(H, 4, s); n = []
        for u in (np.arange(16) + 0.5) / 16:
            ce, se = np.sqrt(1 - u), np.sqrt(u)
            for d in (np.arange(32) + 0.5) * 2 * np.pi / 32:
                a = np.array([ce, 0, se * np.cos(d), se * np.sin(d)]); b = np.array([0, ce, -se * np.sin(d), se * np.cos(d)])
                n.append(hopf.fibre(m, a, b)[1])
        r.append([float(np.mean(n)), int(np.max(n))])
    out[H] = r; print(f"H{H} T4 pieces/fibre mean(max) " + " ".join(f"{a:.0f}({b})" for a, b in r))
json.dump(out, open("pieces.json", "w"))
