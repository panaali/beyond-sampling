"""ARC protocol: estimate = 1 - sum of enumerated error-cone masses, at a compute budget in forward-pass
equivalents (FLOPs / 5.8k).  Compared with plain sampling at the same compute."""
import sys, json, time
sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '/home/user/clean-neurons/m16/approaches/cone_enum')
import numpy as np
import arc_eval as ae
from cones import Counter, FWD
from walk import walk, est_cone
from common import Wi, Wh, Wo

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 64000
res = {}
models = [('orig', 0, (Wi, Wh, Wo))] + [(f's{s}', s, m) for s in (0.0025, 0.005) for m in ae.noisy_models(6, s)]
for name, s, m in models:
    t0 = time.time()
    tr = 0.953179 if name == 'orig' else ae.true_acc(m, n=4_000_000)
    est, fwd = est_cone(m, budget)
    C = Counter(); C.flops = fwd * FWD; masses = walk.keys
    sd_samp = np.sqrt(tr * (1 - tr) / C.fwd_equiv)
    res.setdefault(str(s), []).append(dict(truth=tr, est=est, cones=len(masses), fwd=C.fwd_equiv, lps=C.lps, sd_sampling=sd_samp))
    print(name, s, f'truth {tr:.5f} est {est:.5f} err {100*(est-tr):+.2f}pp cones {len(masses)} fwd-eq {C.fwd_equiv:.0f} '
          f'sampling sd at equal compute {100*sd_samp:.3f}pp  ({time.time()-t0:.0f}s)', flush=True)
for s in ('0.0025', '0.005'):
    r = res[s]; rmse = np.sqrt(np.mean([(x['est'] - x['truth']) ** 2 for x in r]))
    print(s, 'RMSE', 100 * rmse, 'pp; sampling RMSE at equal compute', 100 * np.sqrt(np.mean([x['sd_sampling'] ** 2 for x in r])), 'pp')
json.dump(res, open(f'eval_noisy_{budget}.json', 'w'), indent=1)
