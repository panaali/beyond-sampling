"""Four protocol-frozen cap controls; source-isolated, no estimator imports."""
import ast, hashlib, json, math, time, traceback, warnings
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.special import betainc
P = Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x):
    with p.open('x') as f: json.dump(x, f, indent=2, allow_nan=False)
def progress(x):
    with (P/'cap_progress.jsonl').open('a') as f: f.write(json.dumps(x, allow_nan=False)+'\n')
def safe(x): return float(x) if math.isfinite(float(x)) else str(float(x))
def run():
    gate=json.loads((P/'cap_gate.json').read_text())
    for rel,h in gate['pins'].items(): assert sha(P/rel)==h,rel
    assert not (P/'cap_results.json').exists() and not (P/'cap_progress.jsonl').exists()
    progress({'state':'started','gate':sha(P/'cap_gate.json')})
    start=time.process_time()
    try:
        src=P/'frozen/prune.py'
        tree=ast.parse(src.read_text()); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cap_bound']
        assert len(nodes)==1
        scope={'np':np,'betainc':betainc}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),scope)
        e=F(1,2**30)
        rays=[np.eye(2),np.eye(3),np.array([[1.,0.,0.],[1.,float(e),0.],[1.,0.,float(e)]]),np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])]
        names=['orthant2','orthant3','narrow3','zero_axis2']; records=[]
        for i,(name,R) in enumerate(zip(names,rays,strict=True)):
            R=R/np.linalg.norm(R,axis=1)[:,None]
            charged=[]
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                value=scope['cap_bound'](SimpleNamespace(R=R),SimpleNamespace(add=charged.append))
            record={'name':name,'rays':R.tolist(),'value':safe(value),'finite':math.isfinite(value),'source_ops':charged,'warnings':[str(w.message) for w in ws]}
            if i<2:
                target=.25 if i==0 else (1-1/math.sqrt(3))/2
                assert abs(value-target)<=5e-15
                record.update(analytic_cap=target,exact_cone_mass=.25 if i==0 else .125)
                assert value+5e-15>=record['exact_cone_mass']
            elif i==2:
                # Exact positive-hull box coordinates: alpha2=x2/e,alpha3=x3/e,
                # alpha1=x1-alpha2-alpha3 >= 1-1/4-1/4 = 1/2.
                assert F(1)-F(1,4)-F(1,4)==F(1,2)
                assert 4+2*(e/4)**2<6
                lower=e*e/F(41472)
                record.update(e=str(e),determinant=str(e*e),box_volume=str(e*e/64),strict_mass_lower=str(lower),below_strict_lower=math.isfinite(value) and F(value)<lower)
            else:
                record['zero_axis']=bool(np.all(R.sum(0)==0))
            records.append(record);progress({'state':'case_completed','name':name})
        result={'gate_sha256':sha(P/'cap_gate.json'),'source_sha256':sha(src),'ast_sha256':hashlib.sha256(ast.dump(nodes[0],include_attributes=False).encode()).hexdigest(),'controls':records,'cpu_seconds':time.process_time()-start,'scope':'four analytic cap calls only; no estimator, mass integration, geometry or model execution'}
        write(P/'cap_results.json',result);progress({'state':'completed','result_sha256':sha(P/'cap_results.json')})
    except BaseException:
        progress({'state':'failed','traceback':traceback.format_exc(),'cpu_seconds':time.process_time()-start});raise
if __name__=='__main__':run()
