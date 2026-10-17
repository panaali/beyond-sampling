"""Exact saved-only extrema and interval-separated comparisons."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,sys,time
HERE=Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)

def absolute_interval(row,field):
    d=row[field];lo,hi=F(d['lower']),F(d['upper']);assert lo<=hi
    return (F(0) if lo<=0<=hi else min(abs(lo),abs(hi)),max(abs(lo),abs(hi)))

def main():
    out=HERE/'summary.json';assert not out.exists();t=time.process_time()
    pin=json.loads((HERE/'summary_pins.json').read_text())
    for name,sha in pin.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
    d=json.loads((HERE/'application.json').read_text());assert d['complete'] and d['cases']==64 and not d['failures']
    assert d['statuses']=={'bounded':64};extrema={}
    for field in ['source_minus_target','denominator_only_minus_target','candidate_minus_target']:
        row=max(d['rows'],key=lambda r:absolute_interval(r,field)[1]);lo,hi=absolute_interval(row,field)
        extrema[field]={'key':row['key'],'absolute_lower':str(lo),'absolute_upper':str(hi),'upper_float_display':float(hi)}
    counts=Counter()
    for row in d['rows']:
        s=absolute_interval(row,'source_minus_target');c=absolute_interval(row,'candidate_minus_target')
        counts['candidate_better' if c[1]<s[0] else 'source_better' if s[1]<c[0] else 'overlap_unresolved']+=1
    controls=json.loads((HERE/'controls.json').read_text());retry=json.loads((HERE/'retry_controls.json').read_text())
    result={'complete':True,'pins':pin,'extrema':extrema,'strict_interval_comparison':dict(counts),
            'max_diagonal_drift':str(max(F(r['max_diagonal_drift']) for r in d['rows'])),
            'unique_controls':12,'control_attempts':15,'control_cpu':controls['cpu_seconds']+retry['cpu_seconds'],
            'application_cpu':d['cpu_seconds'],'summary_cpu':time.process_time()-t,
            'scope':'Exact interval-separated argument-angle comparisons only. Overlap is unresolved, not equality. No mass/model/MC replay, wholemass or speed claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'strict_interval_comparison':dict(counts),'max_abs_displays':{k:v['upper_float_display'] for k,v in extrema.items()},'summary_cpu':result['summary_cpu']}))
if __name__=='__main__':main()
