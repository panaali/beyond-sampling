import json,os,re,numpy as np
from scipy.stats import spearmanr
S="study"
items={it["id"]:it for it in json.load(open(f"{S}/keys/items.json"))}
expl={}
for f in sorted(os.listdir(f"{S}/explain_out")):
    for r in json.load(open(f"{S}/explain_out/{f}")): expl[r["id"]]=r
jk=json.load(open(f"{S}/keys/judge.json")); rat={}
for f in sorted(os.listdir(f"{S}/judge_out")):
    for r in json.load(open(f"{S}/judge_out/{f}")): rat[r["jid"]]=r["ratings"]
def auroc(p,n):
    p,n=np.asarray(p,float),np.asarray(n,float); return ((p[:,None]>n[None])+0.5*(p[:,None]==n[None])).mean()
for kind in ["neuron","random_dir"]:
    P={"top":[],"mid":[],"rand":[],"rand_clean":[]}; ra=[];rr=[]
    lit=[0,0]
    for j in jk:
        if j["expl_of"]!=j["snips_of"] or items[j["snips_of"]]["kind"]!=kind: continue
        e=expl[j["expl_of"]]["explanation"]
        for s,x in zip(j["snippets"],rat[j["jid"]]):
            P[s["bin"]].append(x); ra.append(s["rel_act"]); rr.append(x)
            if s["bin"]=="rand" and s["rel_act"]<0.05: P["rand_clean"].append(x)
            if s["bin"]=="top":
                m=re.search(r"\[\[(.*?)\]\]",s["text"]).group(1).strip()
                lit[1]+=1; lit[0]+= (len(m)>0 and m in e)
    print(kind,"mid vs rand %.3f, mid vs rand(<5%%) %.3f, top vs rand(<5%%) %.3f"%(auroc(P["mid"],P["rand"]),auroc(P["mid"],P["rand_clean"]),auroc(P["top"],P["rand_clean"])),
          "spearman(rating,rel_act over all snippets) %.3f"%spearmanr(rr,ra)[0], "held-top marked token appears verbatim in explanation: %d/%d"%tuple(lit))
