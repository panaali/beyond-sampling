import sys,time,numpy as np; sys.path[:0]=["/content/cn/m16/approaches/gpu_cost_law","/content/cn/m16/approaches/understand_m43","/content/cn/m16/approaches/arc_m43_repro"]
import estG, estT_base as B, torch; from m43 import Ops
rng=np.random.default_rng(0)
for d in (4,5):
    R=rng.standard_normal((300,d,d)); R[:100,0]=R[:100,1]+1e-4*rng.standard_normal((100,d)); R/=np.linalg.norm(R,axis=2,keepdims=True)
    ref=np.array([B.orthant(r,Ops()) for r in R]); estG.DEV[0]="cuda"; got=estG.masses_each(R,d); print(d,"maxdiff",np.abs(ref-got).max())
    R=rng.standard_normal((65536,d,d)); R/=np.linalg.norm(R,axis=2,keepdims=True); estG.masses(R[:2048],d); torch.cuda.synchronize()
    t=time.time(); estG.masses(R,d); torch.cuda.synchronize(); print(d, f"{len(R)/(time.time()-t):.0f}/s")
