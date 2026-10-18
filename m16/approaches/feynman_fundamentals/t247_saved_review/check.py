"""Independent saved-only T247 audit. No NumPy/SciPy/author imports or calls."""
import ast
from bisect import bisect_right
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zipfile

HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_56/review/face_grid_centering_controls')
OUT=AUTHOR/'output'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(v): return F(*map(int,v))
def arrays(p):
    out={}
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            b=z.read(name); assert b[:6]==b'\x93NUMPY'
            ver=b[6:8]; size=2 if ver==b'\x01\x00' else 4
            assert ver in (b'\x01\x00',b'\x02\x00')
            n=int.from_bytes(b[8:8+size],'little'); h=ast.literal_eval(b[8+size:8+size+n].decode('latin1'))
            assert h['descr']=='<f8' and h['fortran_order'] is False and len(h['shape'])==1
            payload=b[8+size+n:]; assert len(payload)==8*h['shape'][0]
            a=[v[0] for v in struct.iter_unpack('<d',payload)]
            assert all(math.isfinite(v) for v in a)
            out[name.removesuffix('.npy')]=a
    return out

def dyad(a):
    ratios=[x.as_integer_ratio() for x in a]
    den=max(d for n,d in ratios)
    return [n*(den//d) for n,d in ratios],den

def independent_law(g,c):
    x,dx=dyad(g);v,dv=dyad(c)
    assert len(x)==len(v) and x.count(0)==1 and v[0]==0 and v[-1]==dv
    assert all(a<b for a,b in zip(x,x[1:])) and all(a<=b for a,b in zip(v,v[1:]))
    # Integration by parts: E[X]=g_max-integral C(x)dx. Author sums interval masses instead.
    area=sum((v[j]+v[j+1])*(x[j+1]-x[j]) for j in range(len(x)-1))
    mean=F(x[-1],dx)-F(area,2*dx*dv)
    zero=x.index(0);central=F(v[zero],dv)
    flat=sum(a==b for a,b in zip(v,v[1:]))
    return dict(central=central,central_gap=central-F(1,2),mean=mean,
        cdf_reflection_max=F(max(abs(v[i]+v[-i-1]-dv) for i in range(len(v))),dv),
        grid_reflection_max=F(max(abs(x[i]+x[-i-1]) for i in range(len(x))),dx),
        flat_intervals=flat,positive_intervals=len(x)-1-flat,intervals=len(x)-1,mass=F(1))

def cumulative(a):
    total=0.0;out=[]
    for x in a: total+=x;out.append(total)
    return out

def main():
    start=time.process_time(); target=HERE/'results.json';assert not target.exists()
    r=json.loads((OUT/'results.json').read_text());p=json.loads((AUTHOR/'protocol.json').read_text());gate=json.loads((AUTHOR/'gate.json').read_text())
    assert r['complete'] is True and r['pass_all'] is True and 'fatal' not in r
    for pins in [p['pins'],gate['pins']]:
        for path,digest in pins.items():assert sha(path)==digest,path
    assert r['protocol_sha256']==sha(AUTHOR/'protocol.json') and r['gate_sha256']==sha(AUTHOR/'gate.json')
    review=json.loads(Path(gate['source_review_path']).read_text());assert sha(gate['source_review_path'])==gate['source_review_sha256']
    assert review['complete'] and review['pass_all'] and review['execution_allowed'] and gate['execution_allowed']
    for path,digest in review['pins'].items():assert sha(path)==digest
    assert all(0<t<=20 for t in r['limits']['cpu']) and all(0<t<=2*1024**3 for t in r['limits']['as'])
    assert r['runtime']['python']==p['python'] and r['runtime']['prefix']==p['prefix']
    assert r['runtime']['numpy']==p['numpy'] and r['runtime']['scipy']==p['scipy']
    names=[v[0] for v in p['synthetics']]+['source_T2'];assert [v['name'] for v in r['rows']]==names
    progress=[json.loads(v) for v in (OUT/'progress.jsonl').read_text().splitlines()]
    expected=[('start',None)]+[(k,n) for n in names[:3] for k in ['attempt','returned','checked']]+[('attempt','source_T2'),('source_returned',None),('returned','source_T2'),('checked','source_T2'),('complete',None)]
    assert [(v['kind'],v.get('name')) for v in progress]==expected and len(progress)==15
    assert all(v.get('passed',True) for v in progress)
    assert all(a['cpu']<=b['cpu'] for a,b in zip(progress,progress[1:]))
    source=r['source_return'];assert sha(OUT/source['archive'])==source['sha256'];early=arrays(OUT/source['archive'])
    reports=[]
    for idx,row in enumerate(r['rows']):
        assert row['passed'] and row['status']=='returned' and sha(OUT/row['archive'])==row['sha256']
        a=arrays(OUT/row['archive']);g=a['g'];d=a['d'];dg=1.0 if idx<3 else float.fromhex(row['dg_hex'])
        assert len(g)==(3 if idx<3 else 160001) and len(d)==len(g)
        if idx<3: assert g==[-1.,0.,1.] and d==p['synthetics'][idx][1]
        else:
            assert set(early)=={'g','d','source_cumulative','original'}
            for k in early:assert early[k]==a[k]
            assert dg==g[1]-g[0] and g[0]==-8 and g[-1]==8
        sums=cumulative(d)
        if idx==3: assert sums==a['source_cumulative']
        c=[v*dg for v in sums];original=[(v-c[0])/(c[-1]-c[0]) for v in c]
        assert original==a['original']
        weights=[(d[j]+d[j+1])/2 for j in range(len(d)-1)]
        assert weights==a['trap_weights']
        tc=[0.]+[v*dg for v in cumulative(weights)]
        assert tc[-1]==row['trap_Z'] and [v/tc[-1] for v in tc]==a['trapezoid']
        rr={'name':row['name']}
        for key in ['original','trapezoid']:
            calc=independent_law(g,a[key]);saved=row[key]
            assert set(calc)==set(saved)
            for k,v in calc.items():assert v==(q(saved[k]) if isinstance(saved[k],list) else saved[k]),(row['name'],key,k)
            rr[key]={k:([str(v.numerator),str(v.denominator)] if isinstance(v,F) else v) for k,v in calc.items()}
        if idx<3:
            assert abs(q(row['original']['central'])-F(*p['synthetics'][idx][2]))<=F(1,2**50)
            assert abs(q(row['trapezoid']['central'])-F(*p['synthetics'][idx][3]))<=F(1,2**50)
        else:
            assert c[-1]==row['source_Z12']==source['Z12'] and row['source_Z23']==source['Z23']==0
            di,dd=dyad(d);assert F(max(abs(x-y) for x,y in zip(di,reversed(di))),dd)==q(row['density_reflection_max'])
            assert q(row['original']['central_gap'])>F(1,640000) and q(row['original']['mean'])<0
            assert abs(q(row['trapezoid']['central_gap']))<F(1,10**10) and abs(q(row['trapezoid']['mean']))<F(1,10**9)
            assert row['source_cumulative_calls']==1 and row['intervals']==160000 and row['probes']==9
            probes=[0.,math.nextafter(0.,1.),.25,math.nextafter(.5,0.),.5,math.nextafter(.5,1.),.75,math.nextafter(1.,0.),1.]
            assert probes==a['probes']
            for key in ['original','trapezoid']:
                cv=a[key];ys=a[key+'_probe'];assert len(ys)==9 and all(x<=y for x,y in zip(ys,ys[1:]))
                for u,y in zip(probes,ys):
                    j=bisect_right(cv,u)-1
                    if j>=len(g)-1: expected_y=F(g[-1])
                    else:
                        assert cv[j+1]>cv[j]
                        expected_y=F(g[j])+(F(u)-F(cv[j]))*(F(g[j+1])-F(g[j]))/(F(cv[j+1])-F(cv[j]))
                    assert abs(F(y)-expected_y)<=F(1,2**45)
            # Only saved rounded integral diagnostics: no erf/CDF recomputation.
            assert row['source_Z_delta_diagnostic']==row['source_Z12']-row['analytic_integral_diagnostic']
            assert row['trap_Z_delta_diagnostic']==row['trap_Z']-row['analytic_integral_diagnostic']
        reports.append(rr)
    calls={'source_face_tables_T2':1,'source_ndtr_arrays':1,'source_exp_arrays':1,'trapezoid_tables':4,'synthetic_original_cumsum':3,'np_interp_arrays':2,'np_interp_points':18,'exact_law_scans':8,'model':0,'face_points':0,'rng':0}
    assert r['calls']==calls
    tree=ast.parse((AUTHOR/'frozen/core.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_face_tables' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_g','_dg') for t in n.targets)]
    assert hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()==r['extracted_ast_sha256']
    result=dict(complete=True,pass_all=True,cases=4,law_records=8,progress_events=15,finite_probe_diagnostics=18,reports=reports,review_cpu=time.process_time()-start,author_cpu=r['cpu_main'],author_calls=0,limitations=['No density/exp/ndtr/erf recomputation','No NumPy interpolation call or finite RNG-law inference','Source table correctness beyond saved scalar expression parity remains source-gate scope','No target/model/accuracy/full-cost claim'],pins={str(path):sha(path) for path in [Path(__file__),AUTHOR/'protocol.json',AUTHOR/'gate.json',OUT/'results.json',OUT/'progress.jsonl']})
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['complete','pass_all','cases','review_cpu']}))
if __name__=='__main__':main()
