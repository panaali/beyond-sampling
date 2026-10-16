"""Per-circle variance and cost on the original model; independent vs orthogonal-frame planes."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/great_circle")
from common import Wi, Wh, Wo
from gcirc import walk, frame_planes
M = (Wi, Wh, Wo); TRUTH = 0.95317945
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
nf = 400
U, V = frame_planes(nf, rng)
t = time.time(); f, p, pc = walk(M, U, V); dt = time.time() - t
np.savez(f"pilot_{sys.argv[1] if len(sys.argv)>1 else 0}.npz", f=f, p=p, pc=pc, U=U, V=V)
print(f"{len(f)} circles in {dt:.1f}s; mean {f.mean():.5f} +- {f.std()/np.sqrt(len(f)):.5f} (truth {TRUTH})")
var_c = f.var(); P = p.mean(); PC = pc.mean()
bern = TRUTH * (1 - TRUTH)
print(f"per-circle var {var_c:.3e}  (Bernoulli {bern:.3e}, ratio {bern/var_c:.1f})  pieces {P:.1f} partial {PC:.1f}")
print(f"MSE gain vs sampling per unit compute: full {bern/(var_c*P):.3f}  partial {bern/(var_c*PC):.3f}")
ff = f.reshape(nf, 5).mean(1)
vf = ff.var()
print(f"frame (5 planes) var {vf:.3e} vs indep 5 circles {var_c/5:.3e}  -> frame gain {var_c/5/vf:.2f}")
