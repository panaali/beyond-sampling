"""Four fixed H8T4 prefix parents; no final cells or Gaussian mass."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import hashlib,json,sys,time,resource
import numpy as np
H=Path(__file__).resolve().parent
C=H.parent/'capture'
sys.path.insert(0,str(C))
from common import load_engine,Observer,cone_load
from trained import frozen_state
from canonical import Ledger,pack


def model_from_archive(path):
    d=json.loads(Path(path).read_text());assert d['complete']
    selected=[r for r in d['models'] if r['name']=='H8T4seed0'];assert len(selected)==1
    r=selected[0];m=r['model'];assert r['complete'] and m['H']==8 and m['T']==4
    D=m['D'];assert type(D) is int and D>0 and D&(D-1)==0
    weights=([F(v,D) for v in m['wi']],[[F(v,D) for v in row] for row in m['wh']],[[F(v,D) for v in row] for row in m['wo']])
    model=tuple(np.asarray(a,dtype=float) for a in weights)
    assert [a.shape for a in model]==[(8,),(8,8),(4,8)]
    assert all(np.isfinite(a).all() for a in model)
    assert list(map(F,model[0]))==weights[0]
    assert [[F(v) for v in row] for row in model[1]]==weights[1]
    assert [[F(v) for v in row] for row in model[2]]==weights[2]
    return weights,model,r['checkpoint_sha256']


def analyze(states,weights,record,start):
    Wi,Wh,Wo=weights;cache={};rows=[];L=Ledger()
    for i,s in enumerate(states):
        if time.process_time()-start>17:raise RuntimeError('17CPU pilot whole-phase analysis cap')
        record(dict(stage='state_attempt',index=i))
        hist=tuple(tuple(g) for g in s['history'])
        assert len(hist)==3 and all(len(g)==8 and all(v in (0,1) for v in g) for g in hist)
        if hist not in cache:cache[hist]=frozen_state(Wi,Wh,hist,4,L)
        exact=cache[hist]
        stored=tuple(tuple(F(float.fromhex(v)) for v in row[:3]) for row in s['state'])
        assert len(stored)==8 and all(len(row)==3 for row in stored)
        row=dict(index=i,parent=s['parent'],truth=s['truth'],depth=s['depth'],history=hist,
                 occurrence_id=[s['parent'],i],exact_state=exact,stored_state=stored,
                 map_equal=exact==stored,max_map_difference=max(abs(a-b) for r,q in zip(exact,stored,strict=True) for a,b in zip(r,q,strict=True)))
        rows.append(row);record(dict(stage='state_result',row=pack(row)))
    groups={}
    for field in ('history','exact_state','stored_state'):
        for scope in ('global','same_truth'):
            bucket=defaultdict(list)
            for i,r in enumerate(rows):bucket[(r['truth'],r[field]) if scope=='same_truth' else r[field]].append(i)
            groups[scope+'_'+field]=list(bucket.values())
    resets={k:[g for g in gs if len({rows[i]['history'] for i in g})>1] for k,gs in groups.items() if k.endswith('_state')}
    return dict(rows=rows,groups=groups,distinct_history_reset_groups=resets,
                summary=dict(prefix_pieces=len(rows),group_counts={k:len(v) for k,v in groups.items()},
                             reset_group_counts={k:len(v) for k,v in resets.items()},
                             stored_maps_equal_exact=sum(r['map_equal'] for r in rows),
                             max_map_difference=max((r['max_map_difference'] for r in rows),default=F(0))),
                formation_counts=dict(L.counts),formation_observed_bits=L.bits,
                unique_history_reconstructions=len(cache),cached_history_reuses=len(rows)-len(cache))


def main():
    out=H/'pilot.json';progress=H/'pilot_progress.jsonl'
    assert not out.exists() and not progress.exists()
    p=json.loads((H/'protocol.json').read_text());assert p['source_gate']!='pending'
    start=time.process_time()
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    result=dict(complete=False,protocol_sha256=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest())
    record(dict(stage='phase_attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        assert p['parents']==[0,1,20,21]
        assert str(Path(sys.executable).resolve())==p['runtime']['python_executable']
        assert np.__version__==p['runtime']['numpy_version']
        assert str(Path(np._core._multiarray_umath.__file__).resolve())==p['runtime']['numpy_extension']
        tick=time.process_time();weights,model,checkpoint=model_from_archive(p['model_archive'])
        assert checkpoint==p['checkpoint_sha256']
        reviewed=json.loads(Path(p['predicate_review']).read_text());assert reviewed['complete']
        rr=[r for r in reviewed['rows'] if r['name']=='H8T4seed0'];assert len(rr)==1
        assert all(rr[0]['checks'].values()) and rr[0]['subsets']==256
        result.update(model=pack(weights),checkpoint_sha256=checkpoint,weight_load_cpu=time.process_time()-tick,rational_weight_decodes=104)
        record(dict(stage='model_binding',model=result['model'],checkpoint_sha256=checkpoint))
        tick=time.process_time();E=load_engine(json.loads((C/'protocol.json').read_text()))
        result['source_import_setup_cpu']=time.process_time()-tick
        zero=json.loads((C/'zero_controls.json').read_text());assert zero['all_pass'] and len(zero['initial_cones']['4'])==96
        O=Observer(E,record);parents=[]
        for parent in p['parents']:
            h=zero['initial_cones']['4'][parent]
            initial=dict(sigma=h['sigma'],truth=h['truth'],n_true=np.array([float.fromhex(x) for x in h['n_true']]),cone=cone_load(E,h['cone']))
            parents.append(O.run_parent(parent,initial,model))
        result.update(parents=parents,observer=O.summary(),states=O.states)
        assert result['observer']['mass_or_final_calls']==0 and all(r['states']>0 for r in parents)
        assert [r['parent'] for r in parents]==p['parents']
        result['analysis']=pack(analyze(O.states,weights,record,start))
        result.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      reused_initial_cones=4,new_initial_cones=0,new_mass_reference_MC_calls=0,
                      scope=p['scope'],source_proxy_omissions=p['source_proxy_omissions'])
        record(dict(stage='complete',summary=result['analysis']['summary']))
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],observer=result['observer'],summary=result['analysis']['summary'])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start)
        record(dict(stage='failure',message=repr(e)))
        out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
