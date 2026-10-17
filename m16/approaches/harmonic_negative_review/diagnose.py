"""Frozen T57 root/tie and partial source-cost audit; author files read-only."""
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
import resource
import time
import types
import numpy as np
from hemisphere import HemisphereSpace
from review import AUTHOR,HERE,load_model
from cones import Cone


def legendre_zero(l):
    return F(0) if l%2 else F((-1)**(l//2)*math.comb(l,l//2),2**l)


def legendre(l,x):
    if l==0:return F(1)
    a,b=F(1),x
    for k in range(1,l):a,b=b,((2*k+1)*x*b-k*a)/(k+1)
    return b


def root_mass(D):
    mass=F(1,4)
    for l in range(1,D+1,2):
        lam=(legendre_zero(l-1)-legendre_zero(l+1))/(2*(2*l+1))
        mass+=(2*l+1)*lam*lam*legendre(l,F(-1,2))
    return 6*mass


def evaluate(model,D,tie_fix,normalize,deadline):
    source=(AUTHOR/'hprop.py').read_text()
    insertion='        Wt = [Wh @ A + np.outer(Wi, np.eye(T)[t]) for (_, _, _, A) in br]'
    assert source.count(insertion)==1
    source=source.replace(insertion,'        _map_branches.append(len(br))\n'+insertion)
    old='                a = L[p] - L[q]\n                s = K2.classify(a)'
    assert source.count(old)==1
    if tie_fix:
        new='''                a = L[p] - L[q]
                if not np.any(a):
                    if q < p:
                        ok = False
                        break
                    continue
                s = K2.classify(a)'''
        source=source.replace(old,new)
    module=types.ModuleType('isolated_harmonic_diagnostic')
    exec(compile(source,'<isolated audit copy>','exec'),module.__dict__)
    counts=defaultdict(int)
    class AuditCone(Cone):
        def classify(self,a):
            counts['classify_calls']+=1
            counts['exact_zero_queries']+=int(not np.any(a))
            return super().classify(a)
        def copy(self):
            c=super().copy()
            out=AuditCone.__new__(AuditCone)
            for field in Cone.__slots__:setattr(out,field,getattr(c,field))
            return out
    module.Cone=AuditCone
    module._map_branches=[]
    start=time.process_time()
    sp=HemisphereSpace(3,D,deadline)
    module.space=lambda n,d:sp
    original_roots=module.roots
    if normalize:
        def normalized_roots(T,D):
            return [(cell,K,c/(6*c[0])) for cell,K,c in original_roots(T,D)]
        module.roots=normalized_roots
    root_total=sum(c[0] for cell,K,c in module.roots(3,D))
    estimate,info=module.estimate(model,D,return_info=True)
    H=len(model[0]); T=3
    missing=sum(module._map_branches)*H*T*(2*H+1)
    return dict(estimate=estimate,root_total=root_total,exact_root_total=str(root_mass(D)),
                cpu_seconds=time.process_time()-start,counts=dict(counts),
                gate_map_input_branches=module._map_branches,
                uncharged_dense_gate_map_add_multiply_ops=missing,
                cut_count=info['ncut'],leaf_count=info['nleaf'],
                independent_counts=dict(sp.counts),audit_source_sha256=hashlib.sha256(source.encode()).hexdigest())


def main():
    started=time.process_time(); deadline=started+29
    output=dict(complete=False,rows=[],constant_controls=[],root_formula=[])
    try:
        for D in (2,4,8,16,32):
            output['root_formula'].append(dict(D=D,total=str(root_mass(D)),value=float(root_mass(D))))
        a,_=load_model(0); b,_=load_model(1)
        rng=np.random.default_rng(0)
        noisy=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in b)
        original=json.loads((HERE/'results.json').read_text())
        for name,model in [('seed0',a),('seed1',b),('s0.02_0',noisy)]:
            for D in (2,4):
                base=next(r for r in original['models'] if r['name']==name and r['D']==D)
                variants={}
                for title,tie_fix,normalize in [('unchanged',False,False),('tie_fix',True,False),('tie_and_root',True,True)]:
                    variants[title]=evaluate(model,D,tie_fix,normalize,deadline)
                output['rows'].append(dict(name=name,D=D,truth=base['stored_truth'],
                         truth_se=base['stored_truth_se'],original_cost_fpe=base['stored_cost_fpe'],variants=variants))
        constant=(np.zeros(2),np.zeros((2,2)),np.zeros((3,2)))
        for D in (2,4):
            for title,tie_fix,normalize in [('unchanged',False,False),('tie_fix',True,False),('tie_and_root',True,True)]:
                row=evaluate(constant,D,tie_fix,normalize,deadline)
                output['constant_controls'].append(dict(D=D,variant=title,truth=1/3,**row))
        output['complete']=True
    except (AssertionError,ValueError,TimeoutError) as exc:
        output['failure']=repr(exc)
    output['cpu_seconds']=time.process_time()-started
    output['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (HERE/'diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','constant_controls')},indent=2))

if __name__=='__main__':main()
