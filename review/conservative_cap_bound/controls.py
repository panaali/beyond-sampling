"""One frozen ten-fixture phase; no author routine or probability integration."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time,traceback
from cap import bound
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def log(row):
    with (P/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def packed(rays):return [[v if math.isfinite(v) else str(v) for v in row] for row in rays]
def eye(d):return [[float(i==j) for j in range(d)] for i in range(d)]
def main():
    gate=load(P/'gate.json')
    for path,h in gate['pins'].items():assert sha(Path(path))==h,path
    assert not (P/'results.json').exists() and not (P/'progress.jsonl').exists()
    log({'stage':'started','gate_sha256':sha(P/'gate.json')});start=time.process_time()
    try:
        old=load(Path(gate['old_cap_result']));assert len(old['controls'])==4
        cases=[(r['name'],r['rays'],{}) for r in old['controls']]
        cases += [('orthant4',eye(4),{}),('orthant5',eye(5),{}),('zero_ray3',[[0.,0.,0.],[1.,0.,0.]],{}),('unsupported6',eye(6),{}),('nonfinite3',[[float('nan'),0.,0.]],{}),('bit_limit3',eye(3),{'maxbits':1})]
        assert [name for name,_,_ in cases]==['orthant2','orthant3','narrow3','zero_axis2','orthant4','orthant5','zero_ray3','unsupported6','nonfinite3','bit_limit3']
        rows=[]
        for name,rays,options in cases:
            log({'stage':'fixture_started','name':name});t=time.process_time();r=bound(rays,**options);q=F(r['bound'])
            assert 0<=q<=F(r['outward'])<=1
            record=dict(name=name,rays=packed(rays),options=options,result=r,cpu_seconds=time.process_time()-t)
            if name.startswith('orthant'):
                d=len(rays);truth=F(1,2**d);assert r['fallback'] is None and q>=truth;record['exact_cone_mass']=str(truth);record['bound_over_mass']=str(q/truth)
            elif name=='narrow3':
                e=F(1,2**30);lower=e*e/41472
                assert r['fallback'] is None and q>lower>0
                R=[[F(v) for v in row] for row in rays]
                assert R==[[F(1),F(0),F(0)],[F(1),e,F(0)],[F(1),F(0),e]]
                record['strict_mass_lower']=str(lower);record['old_cap']=old['controls'][2]['value']
            else:
                expected={'zero_axis2':'zero_axis','zero_ray3':'zero_ray','unsupported6':'unsupported_dimension','nonfinite3':'nonfinite_or_type','bit_limit3':'bit_limit'}[name]
                assert r['fallback']==expected and q==1
            rows.append(record);log({'stage':'fixture_completed','name':name,'result':r})
        result=dict(complete=True,gate_sha256=sha(P/'gate.json'),rows=rows,cpu_seconds=time.process_time()-start,author_cap_calls=0,model_geometry_mass_reference_MC_calls=0,scope='exact represented-cone primitive only; no estimator/cost performance claim')
        with (P/'results.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        log({'stage':'completed','result_sha256':sha(P/'results.json')})
    except BaseException:log({'stage':'failed','traceback':traceback.format_exc(),'cpu_seconds':time.process_time()-start});raise
if __name__=='__main__':main()
