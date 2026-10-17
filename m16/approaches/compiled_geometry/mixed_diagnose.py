"""Diagnose frozen mixed-precision rejection without changing acceptance."""
import json,time
import numpy as np
from benchmark import load,HERE,ROOT
from native import walk

def run():
    start=time.process_time();g=np.random.default_rng(660001).standard_normal((32,10,2));q,r=np.linalg.qr(g);q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:];u,v=q[:,:,0],q[:,:,1]
    base,sha=load(HERE.parent/'face_surplus_review/data/H16_T10_original.pth');cases=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base);cases.append((f'H16_sigma{sigma}',model))
    cases.append(('H2_T10',load(ROOT/'work/checkpoints/H2_T10_seed0.pth')[0]));out={'cases':[]}
    for name,model in cases:
        tick=time.perf_counter();old,os=walk(model,u,v,engine='stable');oldtime=time.perf_counter()-tick
        tick=time.perf_counter();new,ns=walk(model,u,v,engine='mixed');newtime=time.perf_counter()-tick
        pieces=np.array(ns['gate_pieces'])-os['gate_pieces'];row=dict(name=name,max_integral_difference=float(np.max(abs(old-new))),piece_difference=pieces.tolist(),mismatched_planes=np.flatnonzero(pieces).tolist(),stable_values=old.tolist(),mixed_values=new.tolist(),stable_counts=os,mixed_counts=ns,stable_wall=oldtime,mixed_wall=newtime,speed_vs_stable=oldtime/newtime,strict_acceptance=bool(np.all(pieces==0) and np.max(abs(old-new))<=1e-12))
        out['cases'].append(row);out['cpu_seconds']=time.process_time()-start;(HERE/'mixed_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(name,row['max_integral_difference'],len(row['mismatched_planes']),row['speed_vs_stable'],flush=True)
if __name__=='__main__':run()
