"""Read-only mapping of all original canonical references to nearest families."""
import json,time,resource
import numpy as np
import torch
from . import runner as R

def gates(model,x):
    wi,wh,wo=model;h=np.zeros(len(wi));out=[]
    for v in x:
        z=wh@h+wi*v;g=z>0;out.extend(map(int,g));h=np.maximum(z,0)
    return out

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time();ops=R.Ops(15);old=R.families('old',ops);new=R.families('nearest',ops);rows=[];counts=[0]*len(new)
    state=torch.load(R.CHECKPOINT,weights_only=True,map_location='cpu');model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    for i,h in enumerate(old):
        pair=tuple(h['perm'][:2]);x=h['cone'].R.mean(0)
        matches=[j for j,k in enumerate(new) if k['pair']==pair and k['sigma']==h['sigma'] and np.array_equal(k['n_true'],h['n_true']) and min(k['cone'].N@x)>-1e-10]
        assert len(matches)==1;j=matches[0];counts[j]+=1
        a=R.engine._ref(h['cone'],h['n_true'],ops);b=R.engine._ref(new[j]['cone'],new[j]['n_true'],ops);a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
        ga,gb=gates(model,a),gates(model,b)
        top=bool(h['n_true'][pair[0]]==1)
        rows.append(dict(original=i,nearest=j,top_closest=top,reference_direction_gap=float(np.linalg.norm(a-b)),gate_differences=sum(x!=y for x,y in zip(ga,gb)),original_gate_pattern=ga,nearest_gate_pattern=gb))
    assert counts.count(2)==24 and counts.count(1)==48 and len(rows)==96
    assert max(r['reference_direction_gap'] for r in rows if not r['top_closest'])<1e-12
    out=dict(complete=True,rows=rows,parent_counts=counts,changed_top=sum(r['reference_direction_gap']>1e-10 for r in rows if r['top_closest']),changed_gate_patterns=sum(r['gate_differences']>0 for r in rows),cpu=time.process_time()-start,reference_selection_ops_proxy=ops.n,forward_gate_diagnostic_calls=192,no_recipe_change=True)
    (R.HERE/'reference_map.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','parent_counts')}))
if __name__=='__main__':main()
