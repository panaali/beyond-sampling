"""T131 check: exact mode of the fixed estT on zoo(8,4,seed): accuracy and total leaf-cell mass. -> leafsum_s{seed}.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro")]
import estT
from m43 import Ops
from zoo44 import zoo
sd = int(sys.argv[1]); m = zoo(8, 4, sd); LEAF = [0.0]; ce0 = estT._cell_error
def ce(c, g, A, Wo, truth, ops):
    LEAF[0] += estT.mass(c, Ops()); return ce0(c, g, A, Wo, truth, ops)
estT._cell_error = ce
o = Ops(); t0 = time.time(); acc = estT.estT(m, o, mode="exact")
r = dict(acc=acc, leafsum=LEAF[0], ops=o.n, cells=estT.STATS["cells"], sec=time.time() - t0)
json.dump(r, open(os.path.join(HERE, f"leafsum_s{sd}.json"), "w"))
print(f"seed {sd} exact acc {acc:.10f} leafsum-1 {LEAF[0]-1:.2e} ops {o.n:.3g} {r['sec']:.0f}s")
