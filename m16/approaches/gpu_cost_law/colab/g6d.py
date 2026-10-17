import sys,time,numpy as np,torch; sys.path[:0]=["/content/cn/m16/approaches/gpu_cost_law","/content/cn/m16/approaches/understand_m43","/content/cn/m16/approaches/arc_m43_repro","/content/cn/m16"]
import estP, estG, orth6
m=estP.load_model(8,6,"0"); estP._init(m,"exact",0.1)
t=time.time(); R=np.concatenate([estP._half(int(i))[1][6] for i in (65,1930)]); print("geom", len(R), f"{time.time()-t:.1f}s", flush=True)
estG.DEV[0]="cuda"
for ch in (512,2048):
    estG.CHUNK[6]=ch; estG.masses(R[:ch],6); torch.cuda.synchronize()
    t=time.time(); m_,o=estG.masses(R,6); torch.cuda.synchronize(); print(ch, f"{len(R)/(time.time()-t):.0f}/s", m_, o, flush=True)
