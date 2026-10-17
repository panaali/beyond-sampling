"""T65 single b17 refinement with charged b9 reuse and checked T59 batching."""
from collections import defaultdict
import hashlib,json,time,sys
import numpy as np
from curved_transport import (HERE,TRANSPORT,FUND,RankRules,conditional_rules,
    stage_grids,hidden_from,transport,target_probability,nonvanishing_certificate)
from batch_terminal import BatchTerminal


def main():
    started=time.process_time();deadline=started+110
    interaction='--interaction' in sys.argv
    nh,nb= (9,17) if interaction else (5,17)
    prior_specs=((5,17),(9,5)) if interaction else ((5,9),)
    output='interaction_results.json' if interaction else 'nested_results.json'
    data=np.load(HERE/'LOCALPILOT_H2_T7_seed0.npz');model=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))
    T=len(model[2]);labels=nonvanishing_certificate(model)['nonzero_state_output_labels']
    modelhash=hashlib.sha256((HERE/'LOCALPILOT_H2_T7_seed0.npz').read_bytes()).hexdigest()
    sources=[HERE/'curved_transport.py',TRANSPORT/'batch_terminal.py',TRANSPORT/'doob.py',TRANSPORT/'backward.py',FUND/'target_tail.py',FUND/'moments.py',FUND/'tail.py']
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result={'classification':'deterministic nested curved-box collocation; no error certificate',
        'checkpoint_sha256':modelhash,'sources':hashes(),'model_input_samples':0,'hidden_nodes':nh,
        'target_nodes':nb,'angles':16,'rank_nodes':64,'previous_construction_provenance':{}}
    engine=BatchTerminal(model);checks=[];scalarwork=defaultdict(float)
    for j in labels:
        _,boxes=stage_grids(model,j,nh,nb)
        for b in (-1.,0.,1.):
            for q in ((0.,0.),(.25,.5)):
                h=hidden_from(np.array([*q,b]),boxes[-1])
                for count in (0,1):
                    exact,work=target_probability(model,h,b,count,j)
                    batch=engine.probability(h,b,count,j)
                    checks.append(abs(exact-batch))
                    scalarwork['calls']+=1
                    for key,val in work.items():
                        if isinstance(val,(int,float)) and key not in ('partition_mass','target_index','prefix_dimension'):scalarwork[key]+=val
    result['batch_scalar_check']={'comparisons':len(checks),'max_abs_difference':max(checks),
        'scalar_work':dict(scalarwork),'batch_work_included_in_cumulative_report':True}
    assert max(checks)<1e-12
    cache={}
    for oldh,oldb in prior_specs:
        prior_key=f'h{oldh}_b{oldb}';result['previous_construction_provenance'][prior_key]={}
        for j in labels:
            path=HERE/f'T7_curved_h{oldh}_b{oldb}_j{j}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert meta['checkpoint_sha256']==modelhash
            result['previous_construction_provenance'][prior_key][j]=meta
            old=np.load(path)['raw'];grids,boxes=stage_grids(model,j,oldh,oldb)
            for i,x in enumerate(grids[-1][0]):
                for k,y in enumerate(grids[-1][1]):
                    for l,b in enumerate(grids[-1][2]):
                        h=hidden_from(np.array([x,y,b]),boxes[-1])
                        for count in (0,1):
                            key=(j,float(h[0]),float(h[1]),float(b),count)
                            if key in cache:assert cache[key]==float(old[i,k,l,count])
                            cache[key]=float(old[i,k,l,count])
    result['previous_unique_terminal_states']=len(cache);cachehits=0;newcalls=0;tables={};grids={};boxes={}
    try:
        for j in labels:
            grids[j],boxes[j]=stage_grids(model,j,nh,nb);g=grids[j][-1]
            raw=np.zeros(tuple(len(v) for v in g)+(2,))
            for i,x in enumerate(g[0]):
                for k,y in enumerate(g[1]):
                    for l,b in enumerate(g[2]):
                        h=hidden_from(np.array([x,y,b]),boxes[j][-1])
                        for count in (0,1):
                            key=(j,float(h[0]),float(h[1]),float(b),count)
                            if key in cache:p=cache[key];cachehits+=1
                            else:
                                if time.process_time()>deadline:raise TimeoutError('terminal deadline')
                                p=engine.probability(h,float(b),count,j);cache[key]=p;newcalls+=1
                            raw[i,k,l,count]=p
            tables[j]=raw
            path=HERE/f'T7_curved_h{nh}_b{nb}_j{j}.npz';np.savez(path,raw=raw)
            path.with_suffix('.json').write_text(json.dumps({'checkpoint_sha256':modelhash,'sources':result['sources'],
                'target':j,'hidden_nodes':nh,'target_nodes':nb,'cumulative_new_terminal_calls':newcalls,
                'cumulative_cache_hits':cachehits,'cumulative_batch_report':engine.report(),
                'previous_construction_provenance':result['previous_construction_provenance']},indent=2)+'\n')
        stats=defaultdict(float);rank=RankRules(64,stats);b=np.linspace(-1,1,nb)
        rules=conditional_rules(b,T,16,rank,stats);constant=(model[0],model[1],np.zeros_like(model[2]))
        q=np.zeros((nb,2));qwork=defaultdict(float)
        for ib,bv in enumerate(b):
            for count in (0,1):
                q[ib,count],work=target_probability(constant,np.zeros(2),float(bv),count,0);qwork['calls']+=1
                for key,val in work.items():
                    if isinstance(val,(int,float)) and key not in ('partition_mass','target_index','prefix_dimension'):qwork[key]+=val
        estimates={};initials={};wmin=1.;wmax=0.
        for j in labels:
            W=tables[j]/q[None,None,:,:];wmin=min(wmin,float(W.min()));wmax=max(wmax,float(W.max()))
            if wmin < -1e-10 or wmax>1+1e-8:raise ArithmeticError(f'conditional range [{wmin},{wmax}]')
            estimates[j],initials[j]=transport(model,j,grids[j],boxes[j],W,rules,stats)
        cg,cb=stage_grids(model,0,nh,nb);ones=np.ones(tuple(len(g) for g in cg[-1])+(2,))
        control,_=transport(constant,0,cg,cb,ones,rules,stats);assert abs(control-1/T)<1e-12
        result.update(accuracy=sum(estimates.values()),contributions=estimates,initial_sign_values=initials,
            constant_W_error=control-1/T,terminal_W_min=wmin,terminal_W_max=wmax,
            transport_work=dict(stats),rank_tail_work=dict(qwork))
    except (ArithmeticError,ValueError,TimeoutError,AssertionError) as error:
        result.update(failure_type=type(error).__name__,failure=str(error),truncated=True)
    result.update(new_terminal_calls=newcalls,terminal_cache_hits=cachehits,batch_report=engine.report(),
        cpu_seconds_excluding_imports=time.process_time()-started,sources_after=hashes())
    result['source_files_changed_during_run']=result['sources']!=result['sources_after']
    (HERE/output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('accuracy','new_terminal_calls','cpu_seconds_excluding_imports','failure','source_files_changed_during_run')}))

if __name__=='__main__':main()
