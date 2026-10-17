"""Reuse original20 float answers once; new MP80 work only when triggered."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,math,time
from hybrid import rescue
from controls import safe,rational,H

def main():
    out=H/'rescue_controls.json';progress=H/'rescue_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'rescue_protocol.json').read_text())
    for f,h in p['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    original=json.loads(Path(p['saved']).read_text());fixtures=[r for r in original['rows'] if r['kind']=='facet'];baseline=json.loads((H/'controls.json').read_text())['rows'];assert [r['id'] for r in fixtures]==[r['id'] for r in baseline]==p['ids'] and len(fixtures)==20
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(safe(r),allow_nan=False)+'\n')
    rows=[];start=time.process_time()
    for old,base in zip(fixtures,baseline,strict=True):
        record(dict(stage='attempt',id=old['id']));t=time.process_time();a=rescue(old['certificate'],base['candidate']);cpu=time.process_time()-t
        v=a['value'];truth=F(Decimal(old['value200']['value']));valid=v is not None and math.isfinite(v) and a['status'] in ['ok','zero_output','opposite_measure_zero','mp80_ok'];diag=dict(valid=valid,absolute_pass=False,relative_pass=False)
        if valid:
            error=abs(F(v)-truth);rel=error/abs(truth) if truth else None
            diag.update(error=rational(error),relative_error=rational(rel) if rel is not None else None,absolute_pass=error<=F(1,10**12),relative_pass=rel<=F(1,10**8) if rel is not None else v==0.0)
        row=dict(id=old['id'],candidate=a,diagnostics=diag,cpu_seconds=cpu,baseline_diagnostics=base['diagnostics']);rows.append(row);record(dict(stage='result',row=row))
    assert len(rows)==20
    result=dict(complete=True,rows=rows,valid=sum(r['diagnostics']['valid'] for r in rows),absolute_pass=sum(r['diagnostics']['absolute_pass'] for r in rows),relative_pass=sum(r['diagnostics']['relative_pass'] for r in rows),mp80_evaluations=sum(r['candidate']['mp_evaluations'] for r in rows),rational_decodes=sum(r['candidate']['rational_decodes'] for r in rows),reused_float_rows=20,new_direct_float_preparation_reference_angular_model_MC_calls=0,cpu_seconds=time.process_time()-start,scope=p['scope'])
    out.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['valid','absolute_pass','relative_pass','mp80_evaluations','cpu_seconds']}))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'rescue_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
