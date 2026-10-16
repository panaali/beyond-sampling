"""End-to-end deterministic observer estimator, ARC harness interface est(model, budget, rng) -> (acc, evals).
acc = 1 - P(b<=0)*blind_err - err23 - err12   (rank-4+ errors not modelled).
'evals' is reported in forward-pass equivalents: anchors x 3 (forward + two reverse-mode rows of the local map)
+ blind probes + quadrature FLOPs / 6000."""
import sys, json
import numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import _blind_probe
import v3, v4
from quad import grid


def make(nv=1, ng=1, q=2, nq=12, noise=0.0, redges=None, rmid=None, ver=3, clip_top=False):
    from v3 import R_EDGES, R_MID
    re_, rm_ = (R_EDGES, R_MID) if redges is None else (redges, rmid)

    def est(model, budget=None, rng=None):
        kw = dict(clip_top=clip_top) if ver == 4 else {}
        o, anchors = (v4 if ver == 4 else v3).estimate(model, nv, ng, q, nq=nq, noise=noise, redges=re_, rmid=rm_, **kw)
        be, nb = _blind_probe(model)
        acc = 1 - (11 / 1024) * be - o['23'].sum() - o['12'].sum()
        nodes = len(grid(nq, nq, nq)[0])
        flops_quad = 2 * 720 * q * nodes * (2 * q + 36)
        return acc, anchors * 3 + nb + flops_quad / 6000, o['23'].sum(), o['12'].sum(), be
    return est


if __name__ == "__main__":
    from arc_eval import noisy_models
    from common import Wi, Wh, Wo
    tr = {'0.0025': json.load(open('../array_rqmc/results/truth_s0025.json')),
          '0.005': json.load(open('../array_rqmc/results/truth_s005.json'))}
    cfg = dict(nv=int(sys.argv[1]), ng=int(sys.argv[2]), q=int(sys.argv[3]), nq=int(sys.argv[4]), noise=float(sys.argv[5]),
               ver=int(sys.argv[6]) if len(sys.argv) > 6 else 3, clip_top=len(sys.argv) > 7 and sys.argv[7] == '1')
    if len(sys.argv) > 8: 
        a, ev, e23, e12, be = make(**cfg)((Wi, Wh, Wo)); print('orig only', cfg, 'bias %+.3f pp e23 %.4f e12 %.4f evals %.0f' % (100 * (a - 0.95317945), e23, e12, ev)); sys.exit()
    est = make(**cfg)
    rows = []
    a, ev, e23, e12, be = est((Wi, Wh, Wo))
    print('orig', cfg, 'est %.5f truth 0.95318 bias %+.3f pp  evals-eq %.0f  e23 %.4f e12 %.4f blind %.3f' % (a, 100 * (a - 0.95317945), ev, e23, e12, be), flush=True)
    for sig in ('0.0025', '0.005'):
        errs = []
        for k, m in enumerate(noisy_models(6, float(sig))):
            a, ev, e23, e12, be = est(m)
            errs.append(a - tr[sig][k])
            print(sig, k, 'est %.4f truth %.4f err %+.2f pp' % (a, tr[sig][k], 100 * errs[-1]), flush=True)
        print(sig, 'RMSE %.3f pp  mean err %+.3f pp  evals-eq %.0f  sampling RMSE at same compute %.3f pp' % (
            100 * np.sqrt(np.mean(np.square(errs))), 100 * np.mean(errs), ev,
            100 * np.sqrt(np.mean([t * (1 - t) for t in tr[sig]]) / ev)), flush=True)
