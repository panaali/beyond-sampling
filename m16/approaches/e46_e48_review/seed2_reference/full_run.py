"""Frozen full seed2 positive replay. Durable attempts; interrupted work is never silently repeated."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib, importlib.util, json, math, resource, time
import numpy as np
P=Path(__file__).resolve().parent
D=P/'full'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def append(p,d):
    with p.open('a') as f: f.write(json.dumps(d,separators=(',',':'),allow_nan=False)+'\n')
def load_protocol():
    d=json.loads((P/'full_protocol.json').read_text())
    for p,h in d['pins'].items(): assert sha(p)==h,p
    target=Fraction(1,10**7*d['N']);tol=1e-7/d['N']
    if Fraction(tol)>target:tol=math.nextafter(tol,0.)
    assert Fraction(tol)<=target
    pilot=json.loads((P/'pilot.json').read_text());assert pilot['complete'] and pilot['returned']==64
    saved={r['index']:r['candidate'] for r in pilot['rows']};assert len(saved)==64
    return d,target,tol,pilot,saved

def validate_row(row,index,tol,saved):
    assert row['index']==index
    assert row['reused_pilot']==(index in saved)
    c=row['candidate']; assert c['mode'] in ('returned','unresolved'),c
    assert not any(row.get(k) is False or c.get(k) is False for k in ('bound_pass','comparison_pass','unchanged_numerical_result'))
    assert 'failure' not in c and 'failure' not in row
    if index in saved:assert c==saved[index],index
    if c['mode']=='returned':
        assert math.isfinite(c['value']) and c['value']>=0
        assert math.isfinite(c['bound']) and 0<=c['bound']<=tol
        assert row['bound_pass'] is True
    else:assert row['bound_pass'] is None and c.get('reason')
    for key in ('setup_selection_counts',):
        assert all(isinstance(v,int) and v>=0 for v in c['center'][key].values())

def read_rows(proto,tol,saved):
    rows=[]; sessions=[];ph=sha(P/'full_protocol.json')
    for p in sorted(D.glob('session_*.json')):
        s=json.loads(p.read_text());assert s['protocol_sha256']==ph
        assert s['status'] in ('soft_stop','complete'),(str(p),s.get('status'))
        sessions.append(s)
    paths=sorted(D.glob('shard_*.jsonl'))
    for j,p in enumerate(paths):
        assert p.name==f'shard_{j:03}.jsonl'
        events=[json.loads(line) for line in p.read_text().splitlines()]
        assert len(events)%2==0,('incomplete attempt preserved',str(p))
        count=len(events)//2;assert 0<count<=proto['shard_size']
        if j<len(paths)-1:assert count==proto['shard_size']
        for k in range(count):
            attempt,result=events[2*k:2*k+2];i=len(rows)
            assert i<proto['N'] and i//proto['shard_size']==j
            assert attempt==dict(stage='attempt',index=i,protocol_sha256=ph)
            assert result['stage']=='result' and result['protocol_sha256']==ph
            row=result['row'];validate_row(row,i,tol,saved);rows.append(row)
    assert sum(s['rows_added'] for s in sessions)==len(rows)
    return rows,sessions

def main():
    start=time.process_time();proto,target,tol,pilot,saved=load_protocol()
    assert proto['execution_gate'], 'Frozen source review and publication required'
    D.mkdir(exist_ok=True)
    prior,sessions=read_rows(proto,tol,saved)
    if len(prior)==proto['N']:print('All rows already complete; no numerical calls');return
    session_path=D/f'session_{len(sessions):03}.json';assert not session_path.exists()
    state=dict(status='running',protocol_sha256=sha(P/'full_protocol.json'),first_index=len(prior),rows_added=0,
               geometry_evaluations=0,model_evaluations=0,source_mass_evaluations=0,reference_evaluations=0)
    rules=None
    def save():
        state.update(cpu=time.process_time()-start,process_cpu=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rules is not None:state.update(rule_events=rules.events,rule_counts=rules.ledger.dict())
        write(session_path,state)
    save()
    try:
        capture=json.loads((P/'capture.json').read_text());assert capture['complete'] and capture['N']==proto['N']
        arrays=[]
        for row in capture['parents']:
            p=P/'parents'/row['archive'];assert sha(p)==row['sha256']
            with np.load(p,allow_pickle=False) as z:
                assert z.files==['rays'];a=z['rays']
            assert a.shape==(row['N'],4,4) and a.dtype==np.float64 and np.isfinite(a).all();arrays.append(a)
        rays=np.concatenate(arrays);assert rays.shape==(proto['N'],4,4)
        assert hashlib.sha256(rays.tobytes(order='C')).hexdigest()==proto['ordered_ray_sha256']
        spec=importlib.util.spec_from_file_location('seed2_full_adapter',proto['adapter']);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        unused_E,unused_guards,M,old,kernel,rescue,radial=adapter.load()
        M.TOL=tol;rescue.TOLERANCE=target;rules=kernel.Rules()
        state.update(allocation=dict(exact=str(target),floating=tol),load_import_cpu=time.process_time()-start,
                     loaded_and_concatenated_float64_elements=2*len(rays)*16,adapter_cold_setup_unpriced=True)
        save()
        for index in range(len(prior),proto['N']):
            if time.process_time()-start>=proto['soft_cpu']:
                state['status']='soft_stop';break
            path=D/f"shard_{index//proto['shard_size']:03}.jsonl"
            append(path,dict(stage='attempt',index=index,protocol_sha256=state['protocol_sha256']))
            c=saved[index] if index in saved else M.one(rays[index],old,kernel,rescue,radial,rules)
            row=dict(index=index,reused_pilot=index in saved,candidate=c,
                     bound_pass=(math.isfinite(c['bound']) and 0<=c['bound']<=tol) if c['mode']=='returned' else None)
            # Preserve failing candidates before validation; nonfinite JSON becomes explicit failure below.
            try:append(path,dict(stage='result',protocol_sha256=state['protocol_sha256'],row=row))
            except Exception:
                append(path,dict(stage='serialization_failure',index=index,candidate_repr=repr(c)));raise
            validate_row(row,index,tol,saved)
            state['rows_added']+=1
            if (index+1)%proto['shard_size']==0:save()
        else:state['status']='complete'
        save()
    except BaseException as e:
        state.update(status='failure',error=repr(e));save();raise
    print(json.dumps({k:state[k] for k in ('status','first_index','rows_added','cpu','maxrss_kib')}))
if __name__=='__main__':main()
