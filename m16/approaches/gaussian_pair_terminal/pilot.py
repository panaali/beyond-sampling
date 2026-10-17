import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from terminal import PairTerminal
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(rows=[],objects=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((HERE/'terminal.py').read_bytes()).hexdigest(),scope='primitive extension at fixed absolute h,u; some crossed h/u/j states are not jointly reachable')
for prefix in ((-1.,1.),(1.,-1.),(1.,1.)):
    h=np.zeros(2)
    for x in prefix:h=np.maximum(model[1]@h+model[0]*x,0.)
    for label in (0,1):
        obj=PairTerminal(model,h,label)
        for u in (-1.,0.,1.):
            row=dict(prefix=prefix,hidden=h.tolist(),label=label,u=u,jointly_consistent_target=(prefix[label]==u))
            try:
                value,stats=obj.probabilities(u);other,otherstats=obj.probabilities(u,swap=True)
                difference=float(np.max(abs(value-other)))
                row.update(values=value.tolist(),swapped_values=other.tolist(),max_axis_disagreement=difference,stats=stats,swapped_stats=otherstats)
                if difference>1e-10:raise ArithmeticError('trained independent axis control failed')
            except Exception as exc:row['failure']=repr(exc)
            out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
            (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
            if 'failure' in row:raise SystemExit(row['failure'])
        out['objects'].append(dict(prefix=prefix,label=label,stats=dict(obj.stats)))
out['total_cpu_seconds']=time.process_time()-start
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['total_cpu_seconds'],rows=len(out['rows']),max_axis_disagreement=max(r['max_axis_disagreement'] for r in out['rows']),cold_geometry_cpu=sum(o['stats']['construction_cpu_seconds'] for o in out['objects']),query_cpu_including_axis_build=sum(o['stats']['query_cpu_seconds'] for o in out['objects']))))
