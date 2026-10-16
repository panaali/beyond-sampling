import time, numpy as np
from system import *
from common import Wi, Wh, Wo
from rosen import G0
res = {}
for pos in (True, False):
    t = time.time(); C.fwd = 0
    b, a, u = mode_table((Wi, Wh, Wo), positive=pos, iters=6)
    print('pos', pos, 'time', round(time.time() - t), 'evals', C.fwd, flush=True)
    np.savez(f'orig_modes_{int(pos)}.npz', b=b, a=a, u=u)
