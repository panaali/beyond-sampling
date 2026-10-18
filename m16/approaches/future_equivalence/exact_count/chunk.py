"""One fixed remaining-chart chunk, bounded independently; no mass calls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys,time,resource
import numpy as np
from pilot import H,C,model_from_archive,load_engine,Observer,cone_load,pack
from analysis import analyze


def load_model(spec):
    if spec['H']==8:return model_from_archive(spec['archive'])
    assert spec['H']==4
    d=json.loads(Path(spec['archive']).read_text());assert d['complete'] and d['T']==4
    one=lambda x:F(int(x[0]),int(x[1]))
    a=d['model'];weights=(list(map(one,a[0])),[list(map(one,r)) for r in a[1]],[list(map(one,r)) for r in a[2]])
    model=tuple(np.asarray(x,dtype=float) for x in weights)
    assert [x.shape for x in model]==[(4,),(4,4),(4,4)] and all(np.isfinite(x).all() for x in model)
    assert list(map(F,model[0]))==weights[0]
    assert [[F(v) for v in r] for r in model[1]]==weights[1]
    assert [[F(v) for v in r] for r in model[2]]==weights[2]
    return weights,model,d['checkpoint_sha256']


def main():
    assert len(sys.argv)==3 and sys.argv[1] in ('4','8')
    width=int(sys.argv[1]);batch=int(sys.argv[2]);assert 0<=batch<8
    tag=f'chunk_h{width}_{batch}'
    out=H/(tag+'.json');progress=H/(tag+'_progress.jsonl');assert not out.exists() and not progress.exists()
    p=json.loads((H/'full_protocol.json').read_text());assert p['source_gate']!='pending' and p['pilot_review']!='pending'
    start=time.process_time()
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    result=dict(complete=False,H=width,batch=batch,protocol_sha256=hashlib.sha256((H/'full_protocol.json').read_bytes()).hexdigest())
    record(dict(stage='phase_attempt',H=width,batch=batch,protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        assert str(Path(sys.executable).resolve())==p['runtime']['python_executable']
        assert np.__version__==p['runtime']['numpy_version']
        assert str(Path(np._core._multiarray_umath.__file__).resolve())==p['runtime']['numpy_extension']
        remaining=list(range(96)) if width==4 else [i for i in range(96) if i not in (0,1,20,21)]
        assert p['chunks'][str(width)]==[remaining[i:i+12] for i in range(0,len(remaining),12)]
        parents=p['chunks'][str(width)][batch];spec=p['models'][str(width)]
        old=json.loads((H/'pilot.json').read_text());assert old['complete'] and old['observer']['mass_or_final_calls']==0
        assert [r['parent'] for r in old['parents']]==[0,1,20,21]
        tick=time.process_time();weights,model,checkpoint=load_model(spec);assert checkpoint==spec['checkpoint_sha256']
        pred=json.loads(Path(p['predicate_review']).read_text());assert pred['complete']
        rr=[r for r in pred['rows'] if r['name']==spec['name']];assert len(rr)==1 and all(rr[0]['checks'].values()) and rr[0]['subsets']==2**width
        result.update(model=pack(weights),checkpoint_sha256=checkpoint,weight_load_cpu=time.process_time()-tick,rational_weight_decodes=width+width*width+4*width)
        record(dict(stage='model_binding',model=result['model'],checkpoint_sha256=checkpoint))
        tick=time.process_time();E=load_engine(json.loads((C/'protocol.json').read_text()));result['source_import_setup_cpu']=time.process_time()-tick
        zero=json.loads((C/'zero_controls.json').read_text());assert zero['all_pass'] and len(zero['initial_cones']['4'])==96
        O=Observer(E,record);parent_rows=[]
        for parent in parents:
            h=zero['initial_cones']['4'][parent]
            initial=dict(sigma=h['sigma'],truth=h['truth'],n_true=np.array([float.fromhex(x) for x in h['n_true']]),cone=cone_load(E,h['cone']))
            parent_rows.append(O.run_parent(parent,initial,model))
        result.update(parents=parent_rows,observer=O.summary(),states=O.states)
        assert result['observer']['mass_or_final_calls']==0 and all(r['states']>0 for r in parent_rows)
        assert result['observer']['lookahead_calls']==result['observer']['lookahead_points']==0 and E.MODE[0]=='exact'
        assert [r['parent'] for r in parent_rows]==parents
        result['analysis']=pack(analyze(O.states,weights,record,start))
        result.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      reused_initial_cones=len(parents),new_initial_cones=0,new_mass_reference_MC_calls=0,
                      scope=p['scope'],source_proxy_omissions=p['source_proxy_omissions'])
        record(dict(stage='complete',summary=result['analysis']['summary']))
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(H=width,batch=batch,complete=True,cpu=result['cpu_seconds'],observer=result['observer'],summary=result['analysis']['summary'])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)))
        out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
