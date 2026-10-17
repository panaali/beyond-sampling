import sys,time,numpy as np,torch; sys.path[:0]=["/content/cn/m16/approaches/gpu_cost_law","/content/cn/m16/approaches/understand_m43","/content/cn/m16/approaches/arc_m43_repro"]
import estG, orth6
rng=np.random.default_rng(3)
R=rng.standard_normal((4096,6,6)); R=R[:,:1]+0.3*R; R/=np.linalg.norm(R,axis=2,keepdims=True)
estG.DEV[0]="cuda"
for ch in (256,1024,2048):
    estG.CHUNK[6]=ch; torch.cuda.reset_peak_memory_stats(); estG.masses(R[:ch],6); torch.cuda.synchronize()
    t=time.time(); estG.masses(R[:2*ch],6); torch.cuda.synchronize(); print(ch, f"{2*ch/(time.time()-t):.0f}/s", torch.cuda.max_memory_allocated()/1e9,"GB",flush=True)
W=rng.standard_normal((3,6,6)); W=W[:,:1]+1.5*W; W/=np.linalg.norm(W,axis=2,keepdims=True)
m=estG.masses_each(W,6)
for k in range(3):
    Rk=torch.as_tensor(W[k],device="cuda"); Bm=torch.linalg.inv(Rk.T); g=torch.Generator(device="cuda"); g.manual_seed(k); hit=0; n=0
    while n<4e9:
        x=torch.randn(1<<25,6,generator=g,device="cuda",dtype=torch.float64); hit+=int(((x@Bm.T)>=0).all(1).sum()); n+=1<<25
    p=hit/n; print("MC", k, m[k], p, (p-m[k])/np.sqrt(p*(1-p)/n), flush=True)
