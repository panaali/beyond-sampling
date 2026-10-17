"""Frozen T65 hidden/target-coordinate and angle-order diagnosis."""
from collections import defaultdict
import hashlib,json,sys,time
import numpy as np
from curved_transport import (HERE,TRANSPORT,FUND,RankRules,conditional_rules,
    stage_grids,terminal,transport,target_probability,nonvanishing_certificate)


def main():
    started=time.process_time();deadline=started+110;reuse='--reuse-kernels' in sys.argv
    data=np.load(HERE/'LOCALPILOT_H2_T7_seed0.npz');model=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))
    T=len(model[2]);labels=nonvanishing_certificate(model)['nonzero_state_output_labels']
    modelhash=hashlib.sha256((HERE/'LOCALPILOT_H2_T7_seed0.npz').read_bytes()).hexdigest()
    result={'classification':'deterministic anisotropic curved-box collocation; no error certificate',
        'checkpoint_sha256':modelhash,'model_input_samples':0,'labels':labels,'rows':[],
        'uses_previously_charged_kernels':reuse,
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'curved_transport.py',TRANSPORT/'doob.py',TRANSPORT/'backward.py',FUND/'target_tail.py']}}
    cache={}
    for nh,nb in ((5,9),(9,5)):
        clock=time.process_time();work=defaultdict(float);tables={};grids={};boxes={};provenance={}
        try:
            for j in labels:
                grids[j],boxes[j]=stage_grids(model,j,nh,nb)
                path=HERE/f'T7_curved_h{nh}_b{nb}_j{j}.npz';meta=path.with_suffix('.json')
                if reuse:
                    provenance[j]=json.loads(meta.read_text())
                    assert provenance[j]['checkpoint_sha256']==modelhash
                    tables[j]=np.load(path)['raw']
                else:
                    local=defaultdict(float);localclock=time.process_time()
                    tables[j]=terminal(model,j,grids[j][-1],boxes[j][-1],local,deadline,cache)
                    provenance[j]={'checkpoint_sha256':modelhash,'hidden_nodes':nh,'target_nodes':nb,'target':j,
                        'construction_work':dict(local),'construction_cpu_seconds':time.process_time()-localclock,
                        'construction_sources':result['sources']}
                    np.savez(path,raw=tables[j]);meta.write_text(json.dumps(provenance[j],indent=2)+'\n')
                    for key,val in local.items():work[key]+=val
            kernelcpu=time.process_time()-clock
            constant=(model[0],model[1],np.zeros_like(model[2]));b=np.linspace(-1,1,nb)
            q=np.zeros((nb,2));qwork=defaultdict(float)
            for ib,bv in enumerate(b):
                for count in (0,1):
                    q[ib,count],qw=target_probability(constant,np.zeros(2),float(bv),count,0);qwork['calls']+=1
                    for key,val in qw.items():
                        if isinstance(val,(int,float)) and key not in ('partition_mass','target_index','prefix_dimension'):qwork[key]+=val
            for nangles in (8,16):
                if time.process_time()>deadline:raise TimeoutError('transport deadline')
                clock=time.process_time();stats=defaultdict(float);rank=RankRules(64,stats)
                rules=conditional_rules(b,T,nangles,rank,stats);estimates={};initials={};wmin=1.;wmax=0.
                for j in labels:
                    W=tables[j]/q[None,None,:,:];wmin=min(wmin,float(W.min()));wmax=max(wmax,float(W.max()))
                    if wmin < -1e-10 or wmax>1+1e-8:raise ArithmeticError(f'conditional range [{wmin},{wmax}]')
                    estimates[j],initials[j]=transport(model,j,grids[j],boxes[j],W,rules,stats)
                cg,cb=stage_grids(model,0,nh,nb)
                ones=np.ones(tuple(len(g) for g in cg[-1])+(2,))
                control,_=transport(constant,0,cg,cb,ones,rules,stats)
                assert abs(control-1/T)<1e-12
                result['rows'].append({'hidden_nodes':nh,'target_nodes':nb,'angles':nangles,'rank_nodes':64,
                    'terminal_grid_points_per_target':nh*nh*nb,'accuracy':sum(estimates.values()),
                    'contributions':estimates,'initial_sign_values':initials,'constant_W_error':control-1/T,
                    'terminal_W_min':wmin,'terminal_W_max':wmax,'transport_work':dict(stats),
                    'kernel_work':dict(work),'kernel_cpu_seconds':kernelcpu,'kernel_reused':reuse or nangles==16,
                    'kernel_provenance':provenance,'rank_tail_work':dict(qwork),'rank_tail_reused':nangles==16,
                    'transport_cpu_seconds':time.process_time()-clock})
                (HERE/'anisotropic_results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'hidden':nh,'target':nb,'angles':nangles,'accuracy':sum(estimates.values()),
                    'new_terminal_calls':work['new_terminal_calls'],'kernel_cpu':kernelcpu,'transport_cpu':time.process_time()-clock}),flush=True)
        except (ArithmeticError,ValueError,TimeoutError,AssertionError) as error:
            result['rows'].append({'hidden_nodes':nh,'target_nodes':nb,'failure_type':type(error).__name__,
                'failure':str(error),'partial_kernel_work':dict(work)})
            result['truncated']=True;break
    result['new_terminal_unique_states']=len(cache);result['cpu_seconds_excluding_imports']=time.process_time()-started
    (HERE/'anisotropic_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu':result['cpu_seconds_excluding_imports'],'new_unique_terminal_states':len(cache)}))

if __name__=='__main__':main()
