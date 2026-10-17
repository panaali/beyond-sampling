"""Independent positive original-domain quadrature on fixed sensitive triangles.

Frozen stored cone2/order8/node447 from owner diagnostic, plus new shifted
coalescing endpoints. No full T6 integration or timing selection.
"""
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path('/workspace/clean-neurons');sys.path.insert(0,str(ROOT))
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit
from m16.approaches.small_ladder_review.portable_cost.t6.triangle import Engine
from m16.approaches.small_ladder_review.portable_cost.t6.jacobi import outer_rule
SOURCE=Path('/workspace/work/recover-small_h2_56/m16/approaches/fixed_t6_triangle')
spec=importlib.util.spec_from_file_location('review_fixed_triangle',SOURCE/'primitive.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
HERE=Path(__file__).resolve().parent

def original(v,n):
    x,w=leggauss(n);x=(x+1)/2;w=w/2
    u,t=np.meshgrid(x,x,indexing='ij')
    pts=u[...,None]*v[0]+((1-u)*t)[...,None]*v[1]+((1-u)*(1-t))[...,None]*v[2]
    return float(np.sum(w[:,None]*w[None,:]*(1-u)/np.sum(pts*pts,axis=-1)**3))

start=time.process_time();rays=np.array(json.loads((SOURCE.parent/'directed_chebyshev/t6_fixture.json').read_text())['rays'])
y,rest,w=outer_rule(8,Audit());ray=rays[2]
cases=[('stored2_n8_i447',ray[:,3:].T,(y@ray[:,:3].T)[447:448],rest[447:448])]
base=np.array([[.43,.17,.13,.11,.09,.07],[.38,.22,.13,.11,.09,.07],[.41,.17,.15,.11,.09,.07]])
anchor=np.array([.08,.12,.19,.23,.17,.21])
for r in (.001,.0001,0.):cases.append(('shifted_'+str(r),base,((1-r)*anchor)[None,:],np.array([r])))
rows=[]
for name,base,p,r in cases:
    v=p[:,None,:]+r[:,None,None]*base[None,:,:]
    old,oldstats=Engine(Audit()).batch(v)
    new,newstats=mod.evaluate(p,r,mod.prepare(base),Engine(Audit()).positive)
    q64=original(v[0],64);q128=original(v[0],128)
    assert abs(q64-q128)<2e-12*abs(q128)
    relative=abs(new[0]/q128-1)
    assert relative<2e-9
    rows.append(dict(case=name,old=float(old[0]),new=float(new[0]),independent64=q64,independent128=q128,new_relative_difference=relative,
                     old_fallbacks=oldstats['positive_quadrature_fallbacks'],new_fallbacks=newstats['positive_quadrature_fallbacks']))
out=dict(rows=rows,cpu_seconds=time.process_time()-start,scope='Fixed sensitive endpoints; independent positive Duffy rules, no roundoff enclosure or full T6 integration.')
(HERE/'fixed_triangle_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
