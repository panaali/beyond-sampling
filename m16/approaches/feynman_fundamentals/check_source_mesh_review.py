"""Independent elementary T4 CDF and bisection of existing h9 mesh nodes.

Frozen falsifier: basis-placement implementation versus a finite heuristic
failure. No shared mesh import, root solver, transport or new model cases.
"""
import json,time
from pathlib import Path
import mpmath as mp
import torch
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_conditioned_transport/source_mesh_cycle')
start=time.process_time();mp.mp.dps=60
saved=json.loads((SOURCE/'controls.json').read_text())
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',weights_only=True,map_location='cpu')
wi=w['rnn.weight_ih_l0'].numpy().ravel()
Phi=lambda x:(1+mp.erf(x/mp.sqrt(2)))/2
alpha=Phi(-6);Z=1-2*alpha
def G(q):return (4*q**3-3*q**4+8*q)/9
mid=G(mp.mpf('.5'))
def physical(s,sg):
    v=G((Phi(sg*s)-alpha)/Z)
    return (v-mid)/(1-mid) if sg>0 else (mid-v)/mid
rows=[]
for row in saved['grids']:
    assert row['metadata']['labels']==[0,1,2]
    mode=row['mode']
    for d,weight in enumerate(wi):
        sg=1 if weight>0 else -1
        for index,actual in enumerate(row['metadata']['axes'][d]):
            target=mp.mpf(index)/8
            if index==0:s=mp.mpf(0)
            elif index==8:s=mp.mpf(6)
            else:
                lo,hi=mp.mpf(0),mp.mpf(6)
                for _ in range(100):
                    s=(lo+hi)/2
                    p=physical(s,sg) if mode=='physical' else 1-physical(6-s,sg)
                    value=(mp.sqrt(s/6)+p)/2
                    if value<target:lo=s
                    else:hi=s
                s=(lo+hi)/2
            expected=s*abs(mp.mpf(float(weight)))*mp.mpf(float(1+1e-12))
            error=abs(mp.mpf(actual)-expected)
            assert error<mp.mpf('2e-12')
            rows.append(dict(mode=mode,coordinate=d,index=index,error=float(error)))
out=dict(rows=rows,max_absolute_error=max(r['error'] for r in rows),cpu_seconds=time.process_time()-start,
         scope='Independent elementary marginal polynomial and 60-digit bisection; no transport or accuracy validation.')
(HERE/'source_mesh_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
