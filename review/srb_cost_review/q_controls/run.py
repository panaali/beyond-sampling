"""One fixed six-fixture Q parity attempt; no original author-module imports."""
from pathlib import Path
from collections import Counter
import json,hashlib,time,resource,warnings,sys,platform
import numpy as np
import scipy
from trace import load,Ledger,LEDGER_NAME
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def invoke(env,model,x,ledger=None):
    if ledger is not None:env[LEDGER_NAME]=ledger
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');value=env['Q'](model,x,radial=False)
    return value,sorted((w.category.__name__,str(w.message)) for w in ws)
def encode(a):return dict(shape=list(a.shape),dtype=str(a.dtype),hex=[float(v).hex() for v in a.ravel()],bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest())
def main():
    start=time.process_time();out=P/'results.json';progress=P/'progress.jsonl'
    assert not out.exists() and not progress.exists()
    result=dict(complete=False,rows=[],new_trained_model_checkpoint_MC_draw_calls=0)
    def log(x):
        with progress.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
    def save():
        result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write(out,result)
    log(dict(stage='started'))
    try:
        proto=json.loads((P/'protocol.json').read_text());gate=json.loads((P/'gate.json').read_text())
        assert gate['complete'] and gate['protocol_sha256']==sha(P/'protocol.json')
        for path,h in {**proto['pins'],**gate['pins']}.items():assert sha(path)==h,path
        result.update(protocol_sha256=sha(P/'protocol.json'),gate_sha256=sha(P/'gate.json'),runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),scope=proto['scope'])
        source=Path(proto['original_source']).read_text();original,oldtree=load(source);traced,newtree=load(source,True)
        result['extracted_source_sha256']=hashlib.sha256(oldtree.encode()).hexdigest();result['transformed_source_sha256']=hashlib.sha256(newtree.encode()).hexdigest()
        write(P/'trees.json',dict(original=oldtree,observed=newtree));save()
        fixtures=json.loads((P/'fixtures.json').read_text());assert len(fixtures)==6
        for case in fixtures:
            if time.process_time()-start>16:raise RuntimeError('16CPU soft cap; preserve partial, no retry')
            row=dict(name=case['name'],runs=[]);result['rows'].append(row);log(dict(stage='fixture_started',name=case['name']));save()
            model=tuple(np.array(case[k],dtype=np.float64) for k in ('Wi','Wh','Wo'));x=np.array(case['x'],dtype=np.float64)
            assert [v.shape for v in model]==[(16,),(16,16),(10,16)] and x.ndim==2 and x.shape[1]==9
            arrays=[('batch',x)]+[(f'row_{i}',x[i:i+1]) for i in range(len(x))]
            for name,xx in arrays:
                log(dict(stage='call_started',fixture=case['name'],name=name));entry=dict(name=name,input=encode(xx));row['runs'].append(entry);save()
                y,ws=invoke(original,model,xx);entry.update(original=encode(y),original_warnings=ws);save()
                L=Ledger();z,wz=invoke(traced,model,xx,L);entry.update(observed=encode(z),observed_warnings=wz,ledger=L.report());save()
                entry['bit_parity']=y.dtype==z.dtype and y.shape==z.shape and y.tobytes()==z.tobytes();entry['warning_parity']=ws==wz
                m=len(xx);J=sum(r['active'] for r in L.intervals);assert len(L.intervals)==19
                expected=dict(float_array=20*m*16+9*m+J*m*(4*16+4*10+6),dense_2mac=20*m*16**2+4*J*m*16*10,dense_dot_expression=10*m*16*(2*16-1)+2*J*m*10*(2*16-1))
                entry['expected_source_counts']=expected;entry['count_parity']=all(L.counts[k]==v for k,v in expected.items());save()
                assert entry['bit_parity'] and entry['warning_parity'] and entry['count_parity'],entry['name']
                log(dict(stage='call_completed',fixture=case['name'],name=name,J=J))
            batch=row['runs'][0]['original']['hex'];single=[r['original']['hex'][0] for r in row['runs'][1:]]
            row['batch_vs_single_hex_equal']=batch==single
            # Diagnostic only: BLAS call shape can legitimately change low bits.
            if case['name']=='zero_readout_preserved_tie':
                row['preserved_tie_defect']=all(float.fromhex(v)==0 for v in batch);assert row['preserved_tie_defect']
            log(dict(stage='fixture_completed',name=case['name']));save()
        result.update(complete=True,fixture_count=6,paired_evaluations=sum(len(r['runs']) for r in result['rows']),all_bit_parity=True,all_warning_parity=True,all_source_count_parity=True,complete_native_cost=False,full_accuracy_certificate=False)
        save();log(dict(stage='completed',results_sha256=sha(out)));print(json.dumps({k:result[k] for k in ('complete','fixture_count','paired_evaluations','cpu_seconds')}))
    except BaseException as e:
        result['failure']=repr(e);save();log(dict(stage='failure',error=repr(e)));raise
if __name__=='__main__':main()
