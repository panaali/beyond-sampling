"""Frozen544 exact-prepared cells; no preparation/angular/model replay."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
from collections import Counter
import hashlib,json,math,time
from hybrid import evaluate
from controls import safe,rational,H

def error(value,truth,absolute):
    if value is None or not math.isfinite(value):return dict(valid=False,absolute_pass=False,relative_pass=False)
    e=abs(F(value)-truth);rel=e/abs(truth) if truth else None
    return dict(valid=value>=0,error=rational(e),relative_error=rational(rel) if rel is not None else None,absolute_pass=value>=0 and e<=absolute,relative_pass=value>=0 and (rel<=F(1,10**8) if rel is not None else value==0.0))

def main():
    out=H/'trained.json';progress=H/'trained_progress.jsonl';assert not out.exists() and not progress.exists()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(safe(r),allow_nan=False)+'\n')
    record(dict(stage='pins_attempt'));p=json.loads((H/'trained_protocol.json').read_text())
    for f,h in p['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    controls=json.loads((H/'rescue_controls.json').read_text());assert controls['complete'] and controls['valid']==controls['absolute_pass']==controls['relative_pass']==20
    old=json.loads(Path(p['saved']).read_text());assert old['complete'] and old['all_checks_pass'] and old['model']=='H16T4seed3'
    fixtures=old['rows'];keys=[(r['sector'],r['unit']) for r in fixtures];assert len(keys)==544 and keys==[(s,u) for s in range(34) for u in range(16)]
    rows=[];start=time.process_time();source_counts=Counter();prep_counts=Counter()
    for r in fixtures:
        record(dict(stage='attempt',sector=r['sector'],unit=r['unit']));t=time.process_time();a=evaluate(r['facet']);cpu=time.process_time()-t
        truth=F(Decimal(r['value200']['value']));diag=error(a['value'],truth,F(1,10**12));valid_status=a['status'] in ['ok','zero_output','opposite_measure_zero','mp80_ok']
        if not valid_status:diag.update(valid=False,absolute_pass=False,relative_pass=False)
        base=a['base'];base_error=error(base['value'],truth,F(1,10**12));source_counts.update(base['counts']);prep_counts.update(r['facet']['counts'])
        sumabs=sum((abs(F(t['contribution'])) for t in base['terms'] if math.isfinite(t['contribution'])),F(0))
        diag['finite_base_term_sumabs']=rational(sumabs)
        row=dict(sector=r['sector'],unit=r['unit'],candidate=a,diagnostics=diag,baseline_diagnostics=base_error,candidate_cpu_seconds=cpu)
        rows.append(row);record(dict(stage='result',row=row))
    assert len(rows)==544 and dict(prep_counts)==old['counts']['facet_exact_named']
    means=[]
    for unit in range(16):
        selected=[r for r in rows if r['unit']==unit];assert [r['sector'] for r in selected]==list(range(34));truth=sum((F(Decimal(fixtures[s*16+unit]['value200']['value'])) for s in range(34)),F(0));published=F(Decimal(old['summary']['mean'][unit]));reference_rounding=abs(truth-published)
        assert reference_rounding<F(1,10**180)
        valid=all(r['diagnostics']['valid'] for r in selected);total=0.0 if valid else None
        if valid:
            for r in selected:total=total+r['candidate']['value']
        diag=error(total,truth,F(1,10**10));row=dict(unit=unit,value=total,reference_exact_saved_sum=rational(truth),published_mean=old['summary']['mean'][unit],reference_sum_discrepancy=rational(reference_rounding),diagnostics=diag);means.append(row);record(dict(stage='mean_result',row=row))
    result=dict(complete=True,model=old['model'],rows=rows,means=means,cell_valid=sum(r['diagnostics']['valid'] for r in rows),cell_absolute_pass=sum(r['diagnostics']['absolute_pass'] for r in rows),cell_relative_pass=sum(r['diagnostics']['relative_pass'] for r in rows),baseline_absolute_pass=sum(r['baseline_diagnostics']['absolute_pass'] for r in rows),baseline_relative_pass=sum(r['baseline_diagnostics']['relative_pass'] for r in rows),mean_absolute_pass=sum(r['diagnostics']['absolute_pass'] for r in means),mean_relative_pass=sum(r['diagnostics']['relative_pass'] for r in means),mp80_evaluations=sum(r['candidate']['mp_evaluations'] for r in rows),rational_decodes=sum(r['candidate']['rational_decodes'] for r in rows),direct_float_evaluations=544,direct_named_partial_counts=dict(source_counts),inherited_preparation_counts=old['counts'],new_exact_preparation_reference_angular_model_MC_calls=0,cpu_seconds=time.process_time()-start,candidate_timers_sum=sum(r['candidate_cpu_seconds'] for r in rows),scope=p['scope'])
    out.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['cell_valid','cell_absolute_pass','cell_relative_pass','mean_absolute_pass','mean_relative_pass','mp80_evaluations','cpu_seconds']}))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'trained_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
