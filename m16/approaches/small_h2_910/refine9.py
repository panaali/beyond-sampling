"""One nested terminal refinement with a hard per-cycle CPU deadline."""
from collections import defaultdict
import hashlib
import json
import pathlib
import time
import numpy as np
from backward import load,invariant_box,possible_labels,target_probability,target_kernel_grid,integrate_backwards,FUND
from doob import RankRules,conditional_rules,transport

HERE=pathlib.Path(__file__).resolve().parent


def main():
    started=time.process_time();deadline=started+110
    model=load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    bound=invariant_box(*model[:2]);labels=possible_labels(model[2])
    grid=(np.linspace(0,bound[0],9),np.linspace(0,bound[1],9),np.linspace(-1,1,9))
    coarse=(np.linspace(0,bound[0],5),np.linspace(0,bound[1],5),np.linspace(-1,1,5))
    assert all(np.array_equal(f[::2],c) for f,c in zip(grid,coarse))
    data=np.load(HERE/'terminal_grid5.npz');cache={};stats=defaultdict(float)
    for j in labels:
        values=np.full((9,9,9,2),np.nan);values[::2,::2,::2,:]=data[f'j{j}']
        cache[j]=values
    data.close()
    output={'grid':9,'reused_grid5_terminal_calls':1000,'new_terminal_calls_planned':4832,
        'target_kernel_sha256':hashlib.sha256((FUND/'target_tail.py').read_bytes()).hexdigest(),
        'rows':[],'complete':False,'model_forward_inputs':0}
    def save():
        np.savez(HERE/'terminal_grid9.npz',**{f'j{j}':v for j,v in cache.items()})
        output['new_terminal_work']=dict(stats)
        output['cpu_seconds_excluding_imports']=time.process_time()-started
        (HERE/'refine9_results.json').write_text(json.dumps(output,indent=2)+'\n')
    try:
        for j in labels:
            values=cache[j]
            for ix,x in enumerate(grid[0]):
                for iy,y in enumerate(grid[1]):
                    for ib,b in enumerate(grid[2]):
                        for count in (0,1):
                            if np.isfinite(values[ix,iy,ib,count]): continue
                            if time.process_time()>deadline:raise TimeoutError('110CPU-second cycle deadline')
                            stats['terminal_attempts']+=1
                            p,work=target_probability(model,np.array([x,y]),float(b),count,j)
                            values[ix,iy,ib,count]=p;stats['terminal_calls']+=1
                            for key,value in work.items():
                                if isinstance(value,(int,float)) and key not in ('partition_mass','prefix_dimension','target_index'):
                                    stats['terminal_'+key]+=value
                            stats['max_partition_error']=max(stats['max_partition_error'],abs(work.get('partition_mass',1)-1))
                save()
        output['new_terminal_cpu']=time.process_time()-started
        constantstats=defaultdict(float)
        actual_constant=target_kernel_grid((model[0],model[1],np.zeros_like(model[2])),0,grid,constantstats,deadline,constant=True)
        exact_q=actual_constant[0,0,:,:]
        rule_stats=defaultdict(float);rank=RankRules(64,rule_stats)
        approximate_q=rank.mass(grid[2],8,2);rules=conditional_rules(grid[2],10,16,rank,rule_stats)
        output['extra_constant_terminal_work']=dict(constantstats)
        output['rank_rule_work']=dict(rule_stats)
        output['max_terminal_q_difference']=float(np.max(abs(exact_q-approximate_q)))
        for representation in ('unconditioned_V','W_approximate_terminal_q','W_exact_terminal_q'):
            for method in ('linear','cubic'):
                if time.process_time()>deadline:raise TimeoutError('110CPU deadline before transport')
                clock=time.process_time();work=defaultdict(float);contributions={}
                for j in labels:
                    if representation=='unconditioned_V':
                        contributions[j]=integrate_backwards(model,j,grid,cache[j],16,work,method=method)[0]
                    else:
                        q=approximate_q if representation=='W_approximate_terminal_q' else exact_q
                        contributions[j]=transport(model,j,grid,cache[j]/q[None,None,:,:],rules,work,method=method)[0]
                controlwork=defaultdict(float)
                if representation=='unconditioned_V':
                    control=integrate_backwards(model,0,grid,actual_constant,16,controlwork,method=method)[0]
                else:
                    control=transport(model,0,grid,actual_constant/q[None,None,:,:],rules,controlwork,method=method)[0]
                output['rows'].append({'representation':representation,'method':method,
                    'accuracy':sum(contributions.values()),'contributions':contributions,'work':dict(work),
                    'constant_control':control,'constant_control_error':control-.1,
                    'control_work':dict(controlwork),'cpu_seconds':time.process_time()-clock})
                save()
        output['complete']=True
    except (ArithmeticError,ValueError,TimeoutError) as error:
        output['failure']=str(error)
    save()
    print(json.dumps(output))

if __name__=='__main__':main()
