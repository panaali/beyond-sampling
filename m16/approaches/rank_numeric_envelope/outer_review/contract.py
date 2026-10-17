"""Apply independently reviewed small-outer source envelopes to saved calls."""
from pathlib import Path
import ast,hashlib,json,time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')

def calls(phase):
    n=phase['nested'];assert phase['overflow']==0
    assert n[0]==n[2] and n[3]==0 and n[4]==n[6] and n[7]==0
    q3=0
    for r in phase['kernels']:
        if r['kernel']=='dlasq3':
            assert not r['query'] and r['failures']==0 and 1<=r['m']<=4
            if r['m']==4:q3+=r['calls']
    # The source callgraph proves small N; n[1]/n[5] are NOT used as maxima.
    return [n[0],n[4],q3]

def price(c):
    q1,q2,q34=c
    return dict(q1_calls=q1,q2_calls=q2,q3_entry_l4=q34,
                q1_exclusive_or_small_helper=28*q1,q1_two_scales=119*q2,
                q2_exclusive_and_small_helpers=426*q2,q2_split_checks=11*q34,
                outer_source_upper=28*q1+545*q2+11*q34)

def main():
    out=HERE/'contraction.json';assert not out.exists()
    controls=json.loads((HERE/'helper_controls.json').read_text())
    assert controls['cases']==630 and controls['failures']==0
    start=time.process_time();summary=json.loads((CENSUS/'summary.json').read_text());pins={};modes={}
    for mode in ['exact','recipe108']:
        geo=[0,0,0];cold=None
        for i in range(8):
            p=CENSUS/f'{mode}_{i}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text())
            geo=[a+b for a,b in zip(geo,calls(d['native']))];c=calls(d['cold'])
            if cold is None:cold=c
            else:assert cold==c
        total=[a+b for a,b in zip(geo,cold)];s=summary['modes'][mode]
        assert total[0]==s['q1_calls'] and total[1]==s['q2_calls']
        assert total[2]==sum(r['calls'] for r in s['rows'] if r['kernel']=='dlasq3' and r['m']==4)
        assert all(min(ast.literal_eval(shape))<=4 for shape in s['rank_shapes'])
        modes[mode]={k:price(v) for k,v in dict(geometry=geo,cold_once=cold,total=total).items()}
    result=dict(modes=modes,pins=pins,cpu_seconds=time.process_time()-start,
        premises='Pinned source/ABI/binding and cold callgraph; all saved Q1/Q2 INFO0; dimension proof is four-column rank plus selected DSYEVD values-only path, NOT lastN.',
        scope='Exclusive named scalar source upper; includes Q1 scales and Q2 small sort/IEEE helpers, excludes Q3/Q4/Q5 and matrix reductions. One cold setup. Integer/control/memory/runtime unpriced; finite helper parity is not all-input compiled equivalence. Not complete rank cost.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:x['total'] for m,x in modes.items()}))
if __name__=='__main__':main()
