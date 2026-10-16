import json,os,numpy as np
from collections import defaultdict
S="study"
items={it["id"]:it for it in json.load(open(f"{S}/keys/items.json"))}
expl={}
batch_of={}
for f in sorted(os.listdir(f"{S}/explain_out")):
    for r in json.load(open(f"{S}/explain_out/{f}")):
        expl[r["id"]]=r; batch_of[r["id"]]=f
jk=json.load(open(f"{S}/keys/judge.json"))
rat={}
for f in sorted(os.listdir(f"{S}/judge_out")):
    for r in json.load(open(f"{S}/judge_out/{f}")): rat[r["jid"]]=r["ratings"]
def auroc(p,n):
    p,n=np.asarray(p,float),np.asarray(n,float)
    if not len(p) or not len(n): return np.nan
    return ((p[:,None]>n[None])+0.5*(p[:,None]==n[None])).mean()
# rel_act of rand snippets by kind
for kind in ["neuron","random_dir"]:
    ra=[s["rel_act"] for j in jk if j["expl_of"]==j["snips_of"] and items[j["snips_of"]]["kind"]==kind for s in j["snippets"] if s["bin"]=="rand"]
    ra=np.array(ra); print(kind,"rand rel_act: mean %.3f median %.3f frac>0.2 %.3f frac>0.05 %.3f"%(ra.mean(),np.median(ra),(ra>0.2).mean(),(ra>0.05).mean()))
    mid=[s["rel_act"] for j in jk if j["expl_of"]==j["snips_of"] and items[j["snips_of"]]["kind"]==kind for s in j["snippets"] if s["bin"]=="mid"]
    print("  mid n",len(mid))
# per-item AUROC
for kind in ["neuron","random_dir"]:
    for a,b in [("top","rand"),("mid","rand")]:
        vals=[]
        for j in jk:
            if j["expl_of"]!=j["snips_of"] or items[j["snips_of"]]["kind"]!=kind: continue
            r=rat[j["jid"]]; by=defaultdict(list)
            for s,x in zip(j["snippets"],r): by[s["bin"]].append(x)
            vals.append(auroc(by[a],by[b]))
        vals=np.array(vals); print(kind,a,b,"per-item mean AUROC %.3f median %.3f nan %d"%(np.nanmean(vals),np.nanmedian(vals),np.isnan(vals).sum()))
# A rate by batch
for f in sorted(set(batch_of.values())):
    ids=[i for i in batch_of if batch_of[i]==f]
    nr=[i for i in ids if items[i]["kind"]=="neuron"]; rd=[i for i in ids if items[i]["kind"]=="random_dir"]
    print(f, "neuronA %d/%d"%(sum(expl[i]["category"]=="A" for i in nr),len(nr)), "randA %d/%d"%(sum(expl[i]["category"]=="A" for i in rd),len(rd)))
# A-rate vs frac_mass, layer, sign
import scipy.stats as st
for kind in ["neuron","random_dir"]:
    ids=[i for i in items if items[i]["kind"]==kind]
    A=np.array([expl[i]["category"]=="A" for i in ids]); fm=np.array([items[i]["frac_mass_above_20"] for i in ids]); L=np.array([items[i]["layer"] for i in ids]); sg=np.array([items[i]["sign"] for i in ids])
    print(kind,"A vs fracmass pointbiserial",st.pointbiserialr(A,fm), "A by layer third",[A[(L>=t*10)&(L<(t+1)*10)].mean().round(2) for t in range(3)], "A by sign +", A[sg==1].mean().round(2), "-",A[sg==-1].mean().round(2), (sg==1).sum())
    conf=np.array([expl[i]["confidence"] for i in ids]); print("  mean conf",conf.mean())
# Fisher exact for H2
from scipy.stats import fisher_exact
print(fisher_exact([[51,33],[10,18]]))
