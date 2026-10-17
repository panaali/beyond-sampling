"""T74 item 4: tiers A/B/C on the 5 trained zoo seeds of M_4,3 (no noise). Truth exact.py (validated by truth_mc.py).
usage: OMP_NUM_THREADS=1 python zoo_clean.py"""
from common_review import zoo
import est_mech as EM
from m43 import Ops
from exact import exact_acc
from run_mech import TIERS
for s in range(5):
    m = zoo(s); ex = exact_acc(m); line = f"zoo seed {s}: exact {ex:.5f}"
    for name in ("A anchor pair, anchor strip", "B default (A2 + resolved crossover)", "C resolved everywhere"):
        kw = dict(TIERS[name]); EM.COMPLETE = kw.pop("complete"); o = Ops(); e = EM.est_mech(m, o, **kw) - ex
        r = e * e / (ex * (1 - ex) / (o.n / 72)); line += f" | {name[0]} err {e:+.4f} ratio {r:.2f} ops {o.n}"
    print(line)
