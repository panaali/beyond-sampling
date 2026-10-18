"""One frozen analytic phase only; no target model or external geometry imports."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import core

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'control_protocol.json'
RESULT=HERE/'controls_result.json'
PROGRESS=HERE/'controls_progress.jsonl'
GATE=HERE/'control_gate.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(obj):
    p=RESULT.with_suffix('.tmp')
    with p.open('w') as f:
        json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(p,RESULT)
def event(obj):
    with PROGRESS.open('a') as f:
        f.write(json.dumps(obj,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def encoded_cone(C):
    return None if C is None else dict(rays=C.rays,normals=C.normals)
def finish_cones(row,children,expected,L):
    row['children']=[encoded_cone(C) for C in children]
    row['expected_rays']=expected
    row['rays_pass']=all((C is None and rays is None) or
        (C is not None and rays is not None and set(C.rays)==set(map(tuple,rays)))
        for C,rays in zip(children,expected,strict=True))
    assert row['rays_pass'],'analytic extreme-ray mismatch'
    row['certificates']=[None if C is None else core.certify_cone(C,L) for C in children]

def primitive(case,row,L):
    if case['kind']=='clip':
        C=core.cone([(1,0,0),(0,1,0),(0,0,1)],[(1,0,0),(0,1,0),(0,0,1)],L)
        children=core.clip(C,case['form'],L)
    else:
        C=core.cone([(1,)],[(1,)],L);children=core.lift(C,case['form'],L)
    row['input_cone']=encoded_cone(C)
    finish_cones(row,children,case['expected_rays'],L)

def network(case,row,L,completed,start):
    wi=case['wi'];wh=case['wh'];D=case['D']
    states=core.initial(wi,wh,D,L);row['depths']=[]
    for depth,expected in enumerate(case['counts'],1):
        assert time.process_time()-start<16,'internal CPU cap'
        if depth>1:states=[child for S in states for child in core.extend(S,wi,wh,D,L)]
        level=dict(depth=depth,count=len(states),expected_count=expected,
                   states=[core.encode(S) for S in states],certificates=[])
        row['depths'].append(level)
        level['unique_histories']=len(set(S.history for S in states))
        level['count_pass']=len(states)==expected==level['unique_histories']
        assert level['count_pass'],'analytic history-count mismatch'
        for S in states:level['certificates'].append(core.certify_state(S,wi,wh,D,L))
    if 'base' in case:
        base=next(r for r in completed if r['id']==case['base'])
        checks=[]
        for actual,old in zip(row['depths'],base['depths'],strict=True):
            oldmap={tuple(map(tuple,s['history'])):s for s in old['states']}
            for s in actual['states']:
                b=oldmap[tuple(map(tuple,s['history']))]
                if case['invariance']=='input_scale2':
                    checks.append(set(s['rays'])==set(b['rays']) and
                        s['hidden']==tuple(tuple(2*x for x in r) for r in b['hidden']))
                else:
                    checks.append(set(s['rays'])=={tuple(-x for x in r) for r in b['rays']} and
                        s['hidden']==tuple(tuple(-x for x in r) for r in b['hidden']))
        row['invariance_checks']=checks;assert all(checks),'exact invariance mismatch'

def invalid(case,row,L):
    try:core.initial(case['wi'],case['wh'],case['D'],L)
    except AssertionError as e:row['rejected']=True;row['exception']=repr(e)
    else:row['rejected']=False
    assert row['rejected'],'invalid API fixture was accepted'

def main():
    assert not RESULT.exists() and not PROGRESS.exists() and not RESULT.with_suffix('.tmp').exists()
    p=json.loads(PROTOCOL.read_text());g=json.loads(GATE.read_text())
    assert g['clearance'] is True and g['protocol_sha256']==sha(PROTOCOL)
    reviewer=Path(g['review_artifact'])
    assert sha(reviewer)==g['review_sha256']
    for name,digest in p['pins'].items():assert sha(HERE/name)==digest,(name,'source drift')
    assert len(p['cases'])==18
    assert resource.getrlimit(resource.RLIMIT_AS)[0]<=2147483648
    assert resource.getrlimit(resource.RLIMIT_CPU)[0]<=20
    start=time.process_time()
    result=dict(status='running',protocol_sha256=sha(PROTOCOL),gate=g,rows=[],
                resource_limits=dict(address_space=resource.getrlimit(resource.RLIMIT_AS),cpu=resource.getrlimit(resource.RLIMIT_CPU)),
                model_calls=0,mass_calls=0,random_calls=0)
    save(result)
    try:
        for case in p['cases']:
            assert time.process_time()-start<16,'internal CPU cap'
            row=dict(id=case['id'],case=case,status='running');L=core.Ledger()
            result['rows'].append(row);save(result);event(dict(phase='attempt',id=case['id']))
            try:
                if case['kind'] in ('clip','lift'):primitive(case,row,L)
                elif case['kind']=='network':network(case,row,L,result['rows'][:-1],start)
                else:invalid(case,row,L)
                row['status']='pass'
            finally:
                row['counts']=L.counts;row['max_bits']=L.max_bits;save(result)
            event(dict(phase='result',id=case['id'],status=row['status']))
        assert len(result['rows'])==18 and all(r['status']=='pass' for r in result['rows'])
        result['status']='complete'
    except BaseException as e:
        result['status']='failure';result['failure']=repr(e);result['traceback']=traceback.format_exc()
        event(dict(phase='failure',error=repr(e)));raise
    finally:
        result['cpu_seconds']=time.process_time()-start
        result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(result)

if __name__=='__main__':main()
