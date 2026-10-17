"""Independent cone decomposition by correct target label; diagnostic only."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
paths={'geometry':Path('/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56/geometry.py'),
       'volume':HERE.parents[1]/'small_h4_4/stable_volume.py'}
modules={}
for name,path in paths.items():
    spec=importlib.util.spec_from_file_location('label_reference_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();g=modules['geometry'].Geometry(seconds=15);cells=g.gates(model);T=len(model[2]);eye=np.eye(T)
out=dict(rows=[],source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for label in range(T):
    pieces=[]
    for vertices,hidden_map in cells:
        logits=model[2]@hidden_map;p=vertices
        for other in range(T):
            if other==label:continue
            normal=logits[label]-logits[other]
            if not np.any(normal) and other<label:p=None;break
            p=g.clip(p,normal)
            if p is None:break
        if p is None:continue
        for above in range(T):
            if above==label:continue
            q=g.clip(p,eye[above]-eye[label])
            if q is None:continue
            for other in range(T):
                if other in (label,above):continue
                q=g.clip(q,eye[label]-eye[other])
                if q is None:break
            if q is not None:pieces.append(q)
    rays=g.simplices(pieces)
    vals=np.array([modules['volume'].mass(ray,20) for ray in rays])
    p,b=(vals.sum(axis=0) if len(vals) else (0.,0.))
    out['rows'].append(dict(label=label,probability=float(p),quadrature_only_bound=float(b),simplices=len(rays)))
out['sum']=sum(r['probability'] for r in out['rows']);out['known_total']=.30773074302392744
out['total_difference']=out['sum']-out['known_total'];assert abs(out['total_difference'])<1e-10
pilot=json.loads((HERE/'pilot.json').read_text());out['comparisons']=[]
for row in pilot['rows']:
    if row['order']!=4:continue
    per_label={str(r['label']):row['stats']['contributions'].get(str(r['label']),0.)-r['probability'] for r in out['rows']}
    out['comparisons'].append(dict(direct=row['direct'],per_label_error=per_label,total_error=sum(per_label.values())))
out['geometry_work']=g.counts;out['volume_nodes']=20**3*sum(r['simplices'] for r in out['rows'])
out['cpu_seconds']=time.process_time()-start
out['scope']='Order20 quadrature bound only; floating geometry/roundoff uncertified, source hashes pinned. Fullspace truth differs from input/target truncated estimator by at most saved omitted masses.'
(HERE/'label_reference.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
