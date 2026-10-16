import json,os,numpy as np,torch
from collections import Counter
from transformers import AutoModel
torch.set_num_threads(4)
A="data/acts"
tokens=np.load(f"{A}/tokens.npy"); dirs=np.load(f"{A}/dirs.npy")
items=json.load(open("study/keys/items.json"))
expl={}
for f in sorted(os.listdir("study/explain_out")):
    for r in json.load(open(f"study/explain_out/{f}")): expl[r["id"]]=r
model=AutoModel.from_pretrained("Qwen/Qwen3-0.6B",dtype=torch.float32).eval()
buf={}
def hook(l):
    def f(m,inp): buf[l]=inp[0].detach()
    return f
for i,L in enumerate(model.layers): L.mlp.down_proj.register_forward_pre_hook(hook(i))
res=[]
for it in [x for x in items if x["kind"]=="random_dir"]:
    L=it["layer"]; d=dirs[L,:,it["dir"]]; s=it["sign"]
    ex=it["show_top"][:12]
    ws=[e[0] for e in ex]; ps=[e[1] for e in ex]
    with torch.inference_mode():
        model.layers=model.layers  # full forward
        model(torch.tensor(tokens[ws],dtype=torch.long))
    h=buf[L].numpy()[np.arange(len(ws)),ps]   # (k,3072)
    c=s*h*d[None]
    proj=c.sum(1)
    top1=c.max(1)/proj
    am=c.argmax(1)
    mc=Counter(am).most_common(1)[0]
    # top-5 share
    top5=np.sort(c,1)[:,-5:].sum(1)/proj
    res.append((it["id"],L,expl[it["id"]]["category"],float(np.median(top1)),float(np.median(top5)),mc[1]/len(ws)))
    print(res[-1],flush=True)
r=np.array([x[3:] for x in res]); cat=np.array([x[2] for x in res])
print("median top1 share %.2f top5 %.2f same-neuron-dominant frac %.2f"%tuple(np.median(r,0)))
for c in "ABCD":
    if (cat==c).any(): print(c,(cat==c).sum(),np.round(r[cat==c].mean(0),2))
