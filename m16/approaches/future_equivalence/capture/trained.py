"""Fixed four-parent prefix capture; exact signatures of represented states only."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import sys,json,gzip,hashlib,time,resource
import numpy as np
from common import H,load_engine,Observer,cone_load
sys.path.insert(0,str(H.parent))
from canonical import canonical,Ledger,pack

def frozen_state(Wi,Wh,history,T,L):
    n=len(Wi);M=[[F(0)]*(T-1) for _ in Wi]
    for t,g in enumerate(history):
        assert len(g)==n and all(x in (0,1) for x in g)
        B=[]
        for i in range(n):
            row=[]
            for j in range(T-1):
                z=F(0)
                for k in range(n):z=L.op('add',z,L.op('mul',Wh[i][k],M[k][j]))
                z=L.op('add',z,Wi[i] if j==t else F(0))
                row.append(L.op('mul',F(g[i]),z))
            B.append(row)
        M=B
    return tuple(map(tuple,M))

def signature(Wi,Wh,Wo,M,L):
    n=len(Wi);d=len(M[0]);B=[]
    for i in range(n):
        row=[]
        for j in range(d):
            z=F(0)
            for k in range(n):z=L.op('add',z,L.op('mul',Wh[i][k],M[k][j]))
            row.append(z)
        B.append(row+[Wi[i]])
    V=[[L.op('sub',Wo[k][i],Wo[0][i]) for i in range(n)] for k in range(1,len(Wo))]
    return canonical(B,V,d+1)

def grouped(rows,field):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['truth'],r['depth'],r[field])].append(i)
    return list(groups.values())

def analyze(states,weights,T,record,start):
    Wi,Wh,Wo=weights;rows=[];L=Ledger();counts=Counter()
    for i,s in enumerate(states):
        if time.process_time()-start>17:raise RuntimeError('17CPU whole-phase analysis cap')
        record(dict(stage='signature_attempt',index=i))
        exact=frozen_state(Wi,Wh,s['history'],T,L)
        stored=tuple(tuple(F(float.fromhex(x)) for x in r[:T-1]) for r in s['state'])
        a=signature(Wi,Wh,Wo,exact,L);b=signature(Wi,Wh,Wo,stored,L)
        counts.update(a['counts']);counts.update(b['counts'])
        row=dict(index=i,parent=s['parent'],truth=s['truth'],depth=s['depth'],history=tuple(map(tuple,s['history'])),
                 exact_state=exact,stored_state=stored,exact_key=a['key'],exact_projective=a['projective'],
                 stored_key=b['key'],stored_projective=b['projective'],map_equal=exact==stored,
                 max_map_difference=max(abs(x-y) for r,q in zip(exact,stored,strict=True) for x,y in zip(r,q,strict=True)),
                 canonical_counts=[a['counts'],b['counts']],canonical_bits=[a['bits'],b['bits']])
        rows.append(row);record(dict(stage='signature_result',row=pack(row)))
    groups={k:grouped(rows,k) for k in ('history','exact_state','stored_state','exact_key','exact_projective','stored_key','stored_projective')}
    extra={k:[g for g in groups[k] if len({rows[i]['exact_state' if k.startswith('exact') else 'stored_state'] for i in g})>1] for k in ('exact_key','exact_projective','stored_key','stored_projective')}
    return dict(rows=rows,groups=groups,additional_different_state_groups=extra,
                summary=dict(states=len(rows),group_counts={k:len(v) for k,v in groups.items()},
                             additional_group_counts={k:len(v) for k,v in extra.items()},
                             stored_maps_equal_exact=sum(r['map_equal'] for r in rows),
                             max_map_difference=max((r['max_map_difference'] for r in rows),default=F(0))),
                formation_counts=dict(L.counts),formation_observed_bits=L.bits,canonical_counts=dict(counts))

def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('3','4');T=int(sys.argv[1])
    out=H/f'trained_t{T}.json';progress=H/f'trained_t{T}_progress.jsonl'
    assert not out.exists() and not progress.exists()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    start=time.process_time();p=json.loads((H/'trained_protocol.json').read_text());assert p['zero_review']!='pending' and p['source_gate']!='pending'
    record(dict(stage='phase_attempt',T=T,protocol_sha256=hashlib.sha256((H/'trained_protocol.json').read_bytes()).hexdigest()))
    result=dict(complete=False,T=T,protocol_sha256=hashlib.sha256((H/'trained_protocol.json').read_bytes()).hexdigest())
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        zero=json.loads((H/'zero_controls.json').read_text());assert zero['all_pass'] and zero['observer']['states']==120
        if T==4:
            prior=json.loads((H/'trained_t3.json').read_text());assert prior['complete'] and prior['observer']['mass_or_final_calls']==0 and prior['protocol_sha256']==result['protocol_sha256']
        tick=time.process_time();E=load_engine(json.loads((H/'protocol.json').read_text()));result['source_import_setup_cpu']=time.process_time()-tick
        tick=time.process_time()
        if T==3:
            with np.load(p['h4t3_npz']) as z:model=tuple(np.array(z[k],dtype=float) for k in ('Wi','Wh','Wo'))
            Wi=list(map(F,model[0]));Wh=[list(map(F,r)) for r in model[1]];Wo=[list(map(F,r)) for r in model[2]]
            checkpoint=p['h4t3_checkpoint_sha256']
        else:
            d=json.loads(gzip.decompress(Path(p['h4t4_geometry']).read_bytes()));decode=lambda z:F(int(z[0]),int(z[1]))
            Wi=list(map(decode,d['model'][0]));Wh=[list(map(decode,r)) for r in d['model'][1]];Wo=[list(map(decode,r)) for r in d['model'][2]]
            checkpoint=d['checkpoint_sha256'];assert checkpoint==p['h4t4_checkpoint_sha256']
            model=(np.array([float(x) for x in Wi]),np.array([[float(x) for x in r] for r in Wh]),np.array([[float(x) for x in r] for r in Wo]))
        assert [a.shape for a in model]==[(4,),(4,4),(T,4)] and all(np.isfinite(a).all() for a in model)
        assert [F(float(x)) for x in model[0]]==Wi
        assert [[F(float(x)) for x in r] for r in model[1]]==Wh and [[F(float(x)) for x in r] for r in model[2]]==Wo
        result.update(model=pack([Wi,Wh,Wo]),checkpoint_sha256=checkpoint,weight_load_cpu=time.process_time()-tick)
        record(dict(stage='model_binding',T=T,model=result['model'],checkpoint_sha256=checkpoint))
        O=Observer(E,record);parents=[]
        for parent in p['parents']:
            h=zero['initial_cones'][str(T)][parent]
            obj=dict(sigma=h['sigma'],truth=h['truth'],n_true=np.array([float.fromhex(x) for x in h['n_true']]),cone=cone_load(E,h['cone']))
            parents.append(O.run_parent(parent,obj,model))
        result.update(parents=parents,observer=O.summary(),states=O.states)
        assert result['observer']['mass_or_final_calls']==0 and all(r['states']>0 for r in parents)
        result['analysis']=pack(analyze(O.states,(Wi,Wh,Wo),T,record,start))
        result.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      reused_initial_cones=4,new_initial_cones=0,new_mass_reference_MC_calls=0,
                      scope=p['scope'],source_proxy_omissions=p['source_proxy_omissions'])
        record(dict(stage='complete',T=T,summary=result['analysis']['summary']));out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(T=T,complete=True,cpu=result['cpu_seconds'],observer=result['observer'],summary=result['analysis']['summary'])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)))
        out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
