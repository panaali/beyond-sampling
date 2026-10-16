"""Check (diagnosis): unbiasedness of each part. MC truth split (4M inputs) vs iid face integrals (n faces).
usage: python check_bias.py <model> n"""
import sys
import numpy as np
import core as C, feyn as F, wtool as W
nm = sys.argv[1]; n = int(sys.argv[2]); M = W.get_model(nm)
rng = np.random.default_rng(99); acc = np.zeros(4)
for _ in range(8):
    x = rng.standard_normal((500_000, 10)); srt = np.sort(x, 1)[:, ::-1]
    err = C.forward(M, x).argmax(1) != np.argsort(x, 1)[:, -2]
    B = srt[:, 1] <= 0; R23 = ~B & (srt[:, 1] - srt[:, 2] < srt[:, 0] - srt[:, 1])
    acc += [err.sum(), (err & B).sum(), (err & R23).sum(), (err & ~B & ~R23).sum()]
mc = acc / 4e6
print(f"MC 4M: err {mc[0]:.5f} B {mc[1]:.5f} R23 {mc[2]:.5f}+-{np.sqrt(mc[2]/4e6):.5f} R12 {mc[3]:.5f}+-{np.sqrt(mc[3]/4e6):.5f}")
reps = []
for r in range(5):
    e23, _ = F.r23(M, n // 360 // 5, "iid", 1000 + r); e12, _ = F.r12(M, n // 45 // 5, "iid", 2000 + r)
    reps.append((e23, e12))
reps = np.array(reps)
print(f"faces: R23 {reps[:,0].mean():.5f}+-{reps[:,0].std()/np.sqrt(5):.5f}  R12 {reps[:,1].mean():.5f}+-{reps[:,1].std()/np.sqrt(5):.5f}")
