"""T136 frozen affine-volume review; Gaussian integration is disabled."""
import time
START = time.process_time()
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import numpy as np
from scipy.spatial import ConvexHull, QhullError

HERE = Path(__file__).resolve().parent
SOURCE = Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review/estT_snapshot.py')
M43 = Path('/workspace/work/admin-t121-rank-mass/m16/approaches/arc_m43_repro/m43.py')
WEIGHTS = Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth')
EXPECTED = {SOURCE: 'afa94627ff5b860a4b922dc340e90d0d6718c770d74e326dfbf667a2bb0bbc3a', M43: 'e4cca6147b5962b709f5d91b0749ed8a25a6f11fdef6b8346b8a16e292e11887', WEIGHTS: 'b771f40fcae826e010b2e89fb698e2ec35026119bc9f66d3ff046dba537b07da'}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

class Ops:
    def __init__(self): self.n = 0
    def add(self, n):
        self.n += n
        if time.process_time()-START > 19:
            raise RuntimeError('19 CPU second review cap')

def hull(X):
    try:
        h = ConvexHull(X)
        return {'volume': float(h.volume), 'vertices': len(h.vertices), 'failure': None}
    except QhullError as e:
        return {'volume': None, 'vertices': None, 'failure': str(e)}

def coordinates(R, a):
    dots = R @ a
    assert np.min(dots) > 0
    Q = np.linalg.qr(a[:, None], mode='complete')[0][:, 1:]
    return (R/dots[:, None] - a/(a@a)) @ Q

def measure(R, a, simplices):
    X = coordinates(R, a)
    raw = hull(X)
    centered = X-X.mean(axis=0)
    _, s, V = np.linalg.svd(centered, full_matrices=False)
    conditioned = {'volume': None, 'failure': 'zero singular value', 'vertices': None}
    if len(s) == 3 and np.min(s) > 0:
        conditioned = hull((centered @ V.T)/s)
        if conditioned['volume'] is not None:
            conditioned['volume'] *= float(np.prod(s))
    v3 = []
    v4 = []
    for r in simplices:
        x = coordinates(r, a)
        v3.append(abs(float(np.linalg.det(x[1:]-x[0])))/6)
        v4.append(float(np.linalg.norm(a))*abs(float(np.linalg.det(r)))/(6*float(np.prod(r@a))))
    total = math.fsum(v3)
    out = dict(raw=raw, conditioned=conditioned, singular_values=s.tolist(),
               singular_ratio=float(s[-1]/s[0]) if s[0] else None,
               source_volume3=total, source_volume4=math.fsum(v4),
               determinant_absolute_discrepancy=math.fsum(abs(x-y) for x,y in zip(v3,v4)),
               rays=len(R), simplices=len(simplices), minimum_anchor_dot=float(np.min(R@a)))
    for name in ('raw', 'conditioned'):
        v = out[name]['volume']
        out[name]['source_minus_hull'] = None if v is None else total-v
        out[name]['relative_difference'] = None if not v else (total-v)/v
    return out

def engine():
    for p, sha in EXPECTED.items():
        assert digest(p) == sha, str(p)
    load('m43', M43)
    return load('t136_pinned_estT', SOURCE)

def source_simplices(E, c, ops):
    result = []
    def record(r, o):
        result.append(r.copy())
        E.STATS['simplices'] += 1
        return 0.
    E.orthant = record
    E.mass(c, ops)
    return result

def box(E, width, rotation):
    R = np.asarray([(x,y,z,1.) for x,y,z in itertools.product((0.,width),(0.,1.),(0.,1.))])
    N = np.array([[1,0,0,0],[-1,0,0,width],[0,1,0,0],[0,-1,0,1],[0,0,1,0],[0,0,-1,1]],float)
    T = R@N.T == 0
    R = R/np.linalg.norm(R,axis=1)[:,None]
    return E.Cone(R@rotation.T, N@rotation.T, T), rotation[:,3]

def controls():
    E = engine(); ops=Ops(); rows=[]
    H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
    orthant=E.Cone.from_normals(np.eye(4))
    cases=[('orthant',orthant,np.ones(4),1/3)]
    for rotname, rot in [('identity',np.eye(4)),('hadamard',H)]:
        for power in (0,10,20,30,45):
            c,a=box(E,2.**-power,rot)
            cases.append((f'{rotname}_width_2^-{power}',c,a,2.**-power))
    for name,c,a,expected in cases:
        row=measure(c.R,a,source_simplices(E,c,ops))
        row.update(name=name,expected_volume=expected,
                   source_relative_error=row['source_volume3']/expected-1)
        for kind in ('raw','conditioned'):
            v=row[kind]['volume']
            row[kind]['expected_relative_error']=None if v is None else v/expected-1
        rows.append(row)
    # Flat geometry is an explicit hull rejection control, not a cone passed to source pulling.
    flat=np.asarray(list(itertools.product((0.,1.),(0.,1.),(0.,))),float)
    result=dict(controls=rows,flat_control=hull(flat),cpu_seconds=time.process_time()-START,
                source_hashes={str(p):digest(p) for p in (*EXPECTED,M43.with_name('m43_weights.npz'),Path(__file__))})
    (HERE/'controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'controls':len(rows),'raw_failures':[r['name'] for r in rows if r['raw']['failure']],
                     'maximum_conditioned_relative_error':max(abs(r['conditioned']['expected_relative_error']) for r in rows),
                     'maximum_source_relative_error':max(abs(r['source_relative_error']) for r in rows),
                     'cpu_seconds':result['cpu_seconds']}))

def trained():
    import torch
    torch.set_num_threads(1)
    E=engine(); E.MODE[0]='exact'; ops=Ops()
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model=(model[0][:,0],model[1],model[2]); halves=E.halves(4)
    records=[]; raw=[]
    out=dict(complete=False,parents=[],leaves=records,random_network_inputs=0,gaussian_integrations=0,
             source_hashes={str(p):digest(p) for p in (*EXPECTED,M43.with_name('m43_weights.npz'),Path(__file__))})
    def save():
        payload={}
        for j,item in enumerate(raw):
            for key,value in item.items(): payload[f'leaf{j}_{key}']=value
        np.savez_compressed(HERE/'raw_leaves.npz',**payload)
        out.update(cpu_seconds_including_imports=time.process_time()-START,source_proxy_ops=ops.n,
                   max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   raw_archive_sha256=digest(HERE/'raw_leaves.npz'))
        (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for index in (0,1,20,21):
        h=halves[index]; parent=h['cone']; a=parent.N.sum(axis=0)
        pm=measure(parent.R,a,source_simplices(E,parent,ops)); begin=len(records)
        def terminal(c,g,A,Wo,truth,o):
            R,N,T=c.R.copy(),c.N.copy(),c.T.copy()
            simplices=source_simplices(E,c,o)
            row=measure(R,a,simplices)
            norms=np.linalg.norm(N,axis=1); safe=np.where(norms>0,norms,1)
            residual=R@N.T/safe
            row.update(parent=index,leaf=len(records),minimum_own_inequality=float(residual.min()),
                       minimum_parent_inequality=float(np.min(R@parent.N.T/np.linalg.norm(parent.N,axis=1))),
                       maximum_symbolic_incidence_residual=float(np.max(abs(residual[T]))) if T.any() else 0.,
                       zero_normals=int(np.sum(norms==0)))
            records.append(row); raw.append(dict(R=R,N=N,T=T,simplices=np.asarray(simplices),anchor=a))
            E.STATS['cells']+=1
            o.add(0)
            return 0.
        E._cell_error=terminal
        E._finish(parent,(h['sigma']*model[0]>0).astype(float),np.outer(model[0],np.eye(4)[0]),1,model,h['truth'],h['n_true'],.1,ops,True)
        leaves=records[begin:]; pv=pm['conditioned']['volume']
        summary=dict(index=index,parent=pm,anchor=a.tolist(),leaves=len(leaves),simplices=sum(r['simplices'] for r in leaves))
        for kind in ('raw','conditioned'):
            good=[r for r in leaves if r[kind]['volume'] is not None]
            diffs=[abs(r[kind]['source_minus_hull']) for r in good]
            summary[kind]=dict(failures=len(leaves)-len(good),hull_sum=math.fsum(r[kind]['volume'] for r in good),
                               sum_absolute_difference=math.fsum(diffs),maximum_absolute_difference=max(diffs,default=0.),
                               summed_screen_pass=len(good)==len(leaves) and math.fsum(diffs)<=1e-9*pv,
                               per_leaf_screen_pass=len(good)==len(leaves) and max(diffs,default=0.)<=1e-10*pv)
            summary[kind]['relative_partition_residual']=summary[kind]['hull_sum']/pv-1
        out['parents'].append(summary); save()
        print(json.dumps(summary),flush=True)
    out['complete']=True; save()

if __name__=='__main__':
    {'controls':controls,'trained':trained}[sys.argv[1]]()
