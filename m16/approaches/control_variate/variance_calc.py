"""Per-model asymptotic variance per unit compute (precise; complements the noisy MSE runs).
For each model, pilot each stratum with fresh samples and compute the exact-design variance:
  sampling   : p(1-p)                         per forward pass
  stratified : sum_h P_h^2 s_h^2 / a_h        (m16/stratified.py design, Y strata)
  rb         : 1.6 Var(Q)
  srb        : 1.6 sum_h P_h^2 sQ_h^2 / a_h   (prefix strata, alloc.json 'plain')
Gain = var_sampling / var_X  (compute-adjusted MSE ratio for unbiased estimators)."""
import sys, os, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,"../.."))
from arc_eval import forward, noisy_models, Wi0, Wh0, Wo0
import stratified as S, cv
from rb9 import COST_Q
level=sys.argv[1]; npil=int(sys.argv[2]) if len(sys.argv)>2 else 10000
d=json.load(open(os.path.join(HERE,"../../stratified_design.json"))); SP,SA=np.array(d["P"]),np.array(d["alloc"])
P=cv.pc()["P"]; A=np.array(json.load(open(os.path.join(HERE,"alloc.json")))["plain"])
models=[(Wi0,Wh0,Wo0)] if level=="0" else noisy_models(6,float(level))
rng=np.random.default_rng(777); out=[]
for m in models:
    sY=[];mY=[]
    for h in range(S.N_STRATA):
        x=S.draw(h,npil,rng); y=(forward(m,x).argmax(1)==np.argsort(x,1)[:,-2]); sY.append(y.std()); mY.append(y.mean())
    p=float(np.dot(SP,mY)); v_s=p*(1-p); v_st=float((SP**2*np.array(sY)**2/SA).sum())
    sQ=[];mQ=[]
    for h in range(cv.NS):
        x=cv.draw(h,npil,rng); q=cv.Q(m,x); sQ.append(q.std()); mQ.append(q.mean())
    mQ=np.array(mQ); sQ=np.array(sQ)
    varQ=float((P*(sQ**2+mQ**2)).sum()-np.dot(P,mQ)**2)
    v_rb=COST_Q*varQ; v_srb=COST_Q*float((P**2*sQ**2/A).sum())
    v_srb_ney=COST_Q*float((P*sQ).sum()**2)
    r=dict(acc=p,gain_strat=v_s/v_st,gain_rb=v_s/v_rb,gain_srb=v_s/v_srb,gain_srb_oracle_neyman=v_s/v_srb_ney,srb_vs_strat=v_st/v_srb)
    print(json.dumps({k:round(v,3) for k,v in r.items()}),flush=True); out.append(r)
json.dump(out,open(os.path.join(HERE,f"variance_{level}.json"),"w"),indent=1)
