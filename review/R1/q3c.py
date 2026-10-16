import json,os,numpy as np,torch,time
from collections import Counter
from transformers import AutoModel
torch.set_num_threads(2)
A="data/acts"; tokens=np.load(f"{A}/tokens.npy"); dirs=np.load(f"{A}/dirs.npy")
items=json.load(open("study/keys/items.json")); expl={}
for f in sorted(os.listdir("study/explain_out")):
    for r in json.load(open(f"study/explain_out/{f}")): expl[r["id"]]=r
model=AutoModel.from_pretrained("Qwen/Qwen3-0.6B",dtype=torch.float32,attn_implementation="eager").eval()
allL=model.layers; buf={}
class Stop(Exception): pass
def hook(m,inp): buf["h"]=inp[0].detach(); raise Stop
rd=sorted([x for x in items if x["kind"]=="random_dir"],key=lambda x:x["layer"])
res=[]
for it in rd:
    L=it["layer"]; d=dirs[L,:,it["dir"]]; s=it["sign"]
    ex=it["show_top"][:8]; ws=[e[0] for e in ex]; ps=[e[1] for e in ex]
    hdl=allL[L].mlp.down_proj.register_forward_pre_hook(hook)
    t0=time.time()
    try:
        with torch.inference_mode(): model(torch.tensor(tokens[ws],dtype=torch.long))
    except Stop: pass
    hdl.remove()
    h=buf["h"].numpy()[np.arange(len(ws)),ps]; c=s*h*d[None]; proj=c.sum(1)
    am=c.argmax(1); mc=Counter(am).most_common(1)[0]
    res.append((it["id"],L,expl[it["id"]]["category"],round(float(np.median(c.max(1)/proj)),2),round(float(np.median(np.sort(c,1)[:,-5:].sum(1)/proj)),2),mc[1]/len(ws)))
    print(res[-1],"chk",np.round(proj[:3]/np.array([e[2] for e in ex[:3]]),3),round(time.time()-t0,1),flush=True)
r=np.array([x[3:] for x in res]); cat=np.array([x[2] for x in res])
print("median top1 %.2f top5 %.2f dominant-same-neuron %.2f"%tuple(np.median(r,0)))
for c in "ABCD":
    if (cat==c).any(): print(c,(cat==c).sum(),np.round(r[cat==c].mean(0),2))
