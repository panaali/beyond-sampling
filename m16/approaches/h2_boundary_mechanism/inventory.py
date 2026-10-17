"""Weight-derived H2 readout and final-input boundary inventory; no model inputs."""
import hashlib,json,sys,time,resource
from pathlib import Path
import numpy as np
BASE=Path(__file__).parent
sys.path.insert(0,str(BASE.parent/'small_h2_56'))
from models import load,noisy
sys.path.pop(0)

def envelope(wo):
    breaks=[0.]
    for i in range(len(wo)):
        for j in range(i):
            a,b=wo[i]-wo[j]
            if b!=0 and -a/b>0:breaks.append(float(-a/b))
    breaks=sorted(set(breaks))+[float('inf')]
    sectors=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        midpoint=(lo+hi)/2 if np.isfinite(hi) else lo+max(1.,abs(lo))
        label=int(np.argmax(wo[:,0]+midpoint*wo[:,1]))
        if sectors and sectors[-1]['label']==label:sectors[-1]['upper']=hi
        else:sectors.append(dict(lower=lo,upper=hi,label=label))
    return sectors

def describe(model):
    wi,wh,wo=model;sectors=envelope(wo);boundaries=[]
    for left,right in zip(sectors[:-1],sectors[1:]):
        c=wo[left['label']]-wo[right['label']]
        denominator=float(c@wi);scale=float(abs(c*wi).sum())
        boundaries.append(dict(left=left['label'],right=right['label'],ratio=left['upper'],margin=c.tolist(),
             last_input_coefficient=denominator,cancellation_ratio=abs(denominator)/scale if scale else 0.,
             previous_state_slope=(-c@wh/denominator).tolist() if denominator else None))
    labels=set(s['label'] for s in sectors)|{int(wo[:,0].argmax()),int(wo[:,1].argmax()),0}
    return dict(weights=[w.tolist() for w in model],sectors=sectors,boundaries=boundaries,
                axis_labels=[int(wo[:,0].argmax()),int(wo[:,1].argmax())],zero_label=0,
                potential_labels=sorted(labels),unavailable_label_error_floor=1-len(labels)/len(wo),
                recurrent_inf_gain=float(abs(wh).sum(axis=1).max()),
                recurrent_two_gain=float(np.linalg.norm(wh,2)),
                input_signs=np.sign(wi).tolist())

if __name__=='__main__':
    start=time.process_time();out={'models':[]}
    for t in (4,5):
        base,sha=load(t)
        for sigma in (0.,.0025,.02):
            model=base if sigma==0 else noisy(base,sigma)
            row=dict(T=t,sigma=sigma,checkpoint_sha256=sha,**describe(model));out['models'].append(row)
            print(t,sigma,'weights',row['weights'],'sectors',row['sectors'],'boundaries',row['boundaries'],flush=True)
    out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (BASE/'inventory.json').write_text(json.dumps(out,indent=2)+'\n')
