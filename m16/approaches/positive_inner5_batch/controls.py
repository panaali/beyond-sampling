"""30 frozen controls, old scalar values reused without scalar execution."""
from pathlib import Path
import hashlib,json,time,sys,math
import numpy as np
from primitive import integrate
H=Path(__file__).resolve().parent

def number(x):return float.fromhex(x) if isinstance(x,str) else float(x)
def safe(x):
    if isinstance(x,float) and not math.isfinite(x):return x.hex()
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    return x

def main():
    out=H/'controls.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((H/'protocol.json').read_text())
    for p,h in P['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    old=json.loads(Path(P['old_results']).read_text());assert old['complete'] and len(old['rows'])==26
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(safe(r),allow_nan=False)+'\n')
    start=time.process_time();record(dict(stage='batch_attempt',ids=[r['id'] for r in old['rows']]))
    ps=np.array([[number(x) for x in r['P']] for r in old['rows']],dtype=np.float64)
    ds=np.array([[number(x) for x in r['D']] for r in old['rows']],dtype=np.float64)
    pbits=ps.tobytes();dbits=ds.tobytes();t=time.process_time();ans=integrate(ps,ds);candidate_cpu=time.process_time()-t
    assert len(ans)==len(old['rows'])==26
    unchanged=ps.tobytes()==pbits and ds.tobytes()==dbits;rows=[]
    for prior,got in zip(old['rows'],ans,strict=True):
        exp=prior['candidate'];ok=got['status']==exp['status'] and got['branch']==exp['branch'];bits=None
        if exp['status']=='ok':bits=float(got['value']).hex()==number(exp['value']).hex();ok=ok and bits
        if 'exception' in exp:ok=ok and got.get('exception')==exp['exception']
        row=dict(id=prior['id'],expected=exp,actual=got,status_branch_pass=got['status']==exp['status'] and got['branch']==exp['branch'],supported_bits_pass=bits,passed=bool(ok and unchanged));rows.append(row);record(row)
    fixtures=[('empty',np.zeros((0,5)),np.zeros((0,5))),('width4',np.zeros((2,4)),np.zeros((2,4))),('different_N',np.zeros((1,5)),np.zeros((2,5))),('float32',np.zeros((1,5),dtype=np.float32),np.zeros((1,5)))]
    guards=[]
    for name,p,d in fixtures:
        record(dict(stage='guard_attempt',name=name));error=None
        try:integrate(p,d)
        except ValueError as e:error=str(e)
        row=dict(id=name,passed=error is not None,error=error);guards.append(row);record(row)
    assert len(rows)==26 and len(guards)==4
    result=dict(complete=True,cases=30,rows=rows,guards=guards,all_pass=all(r['passed'] for r in rows+guards),input_bytes_unchanged=unchanged,candidate_cpu_seconds=candidate_cpu,cpu_seconds=time.process_time()-start,numpy=np.__version__,python=sys.version,old_scalar_reference_mass_model_calls=0,new_batch_primitive_calls=1,new_invalid_guard_calls=4,scope=P['scope'],cost='No new source/native cost or matched speed claim; vector masks/copies/ufunc dispatch/runtime remain unpriced.')
    out.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_pass=result['all_pass'],cases=30,candidate_cpu_seconds=candidate_cpu,cpu_seconds=result['cpu_seconds'])))
    assert result['all_pass'], 'durable controls contain failure'
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
