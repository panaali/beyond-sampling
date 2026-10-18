"""T284 check of fused_gid.py (GPU): (a) gid = arange reproduces fused.py bit for bit; (b) models 10 and 20 of a
30-model pack re-trained alone with gid = [10, 20] are bit-identical to their packed copies. 300 steps, chunks of 100."""
import os, sys
import torch as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused as F, fused_gid as G  # noqa: E402
seeds = list(range(500, 530)); kw = dict(n_train=[2 ** 30] * 30, data_seed=41)
a = F.FusedTrainer(seeds, **kw); b = G.FusedTrainer(seeds, **kw)
c = G.FusedTrainer([seeds[10], seeds[20]], n_train=[2 ** 30] * 2, data_seed=41, gid=[10, 20])
for _ in range(3):
    a.step(100); b.step(100); c.step(100)
th.cuda.synchronize()
for nm in ("Wi", "Wh", "Wo"):
    A, Bt, Cc = getattr(a, nm), getattr(b, nm), getattr(c, nm)
    print(nm, "fused vs gid-arange equal:", bool(th.equal(A, Bt)), "| alone vs packed [10,20] equal:",
          bool(th.equal(Cc, A[[10, 20]])), "max diff %.3g" % float((Cc - A[[10, 20]]).abs().max()))
print("moved from init:", float((a.Wh - G.FusedTrainer(seeds, **kw).Wh).abs().max()))
