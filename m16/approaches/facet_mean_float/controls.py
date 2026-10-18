"""Twenty fixed saved facet controls; no exact preparation or MP replay."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,math,time
from kernel import evaluate
H=Path(__file__).resolve().parent

def safe(x):
    if isinstance(x,float) and not math.isfinite(x):return x.hex()
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    return x

def rational(x):return [str(x.numerator),str(x.denominator)]
def main():
    out=H/'controls.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'protocol.json').read_text())
    for f,h in p['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    old=json.loads(Path(p['saved']).read_text());fixtures=[r for r in old['rows'] if r['kind']=='facet'];assert len(fixtures)==20 and [r['id'] for r in fixtures]==p['ids']
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(safe(r),allow_nan=False)+'\n')
    rows=[];start=time.process_time()
    for r in fixtures:
        record(dict(stage='attempt',id=r['id']));t=time.process_time();a=evaluate(r['certificate']);cpu=time.process_time()-t
        value=a['value'];truth=F(Decimal(r['value200']['value']));valid=value is not None and math.isfinite(value) and a['status'] in ['ok','zero_output','opposite_measure_zero']
        diag=dict(valid=valid,absolute_pass=False,relative_pass=False)
        if valid:
            error=abs(F(value)-truth);scale=abs(truth);relative=error/scale if scale else None
            diag.update(error=rational(error),relative_error=rational(relative) if relative is not None else None,absolute_pass=error<=F(1,10**12),relative_pass=relative<=F(1,10**8) if relative is not None else value==0.0)
            contributions=[F(t['contribution']) for t in a['terms']];sumabs=sum(map(abs,contributions),F(0));diag['saved_term_cancellation_ratio']=rational(sumabs/abs(F(value))) if value else None
        row=dict(id=r['id'],candidate=a,saved_value200=r['value200'],saved_certificate_status=r['certificate']['status'],diagnostics=diag,candidate_cpu_seconds=cpu,inherited_exact_preparation_counts=r['certificate']['counts'],inherited_exact_preparation_bits=r['certificate']['bits'])
        rows.append(row);record(dict(stage='result',row=row))
    assert len(rows)==20
    result=dict(complete=True,rows=rows,valid=sum(r['diagnostics']['valid'] for r in rows),absolute_pass=sum(r['diagnostics']['absolute_pass'] for r in rows),relative_pass=sum(r['diagnostics']['relative_pass'] for r in rows),cpu_seconds=time.process_time()-start,new_float_evaluations=20,new_exact_preparation_MP_angular_model_MC_calls=0,scope=p['scope'])
    out.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['valid','absolute_pass','relative_pass','cpu_seconds']}))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
