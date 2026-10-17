from collections import defaultdict
import json
import pathlib
import time
import numpy as np
from backward import load,invariant_box,possible_labels,integrate_backwards,target_kernel_grid
from doob import RankRules,conditional_rules,transport

HERE=pathlib.Path(__file__).resolve().parent

def main():
    start=time.process_time();model=load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    bound=invariant_box(*model[:2]);labels=possible_labels(model[2])
    result={'uses_previously_charged_terminal_caches':True,'model_forward_inputs':0,'rows':[]}
    for n in (5,7):
        grid=(np.linspace(0,bound[0],n),np.linspace(0,bound[1],n),np.linspace(-1,1,n))
        cache=np.load(HERE/f'terminal_grid{n}.npz')
        cstats=defaultdict(float)
        constant=target_kernel_grid((model[0],model[1],np.zeros_like(model[2])),0,grid,cstats,start+110,constant=True)
        rankstats=defaultdict(float);rank=RankRules(64,rankstats)
        q=rank.mass(grid[2],8,2);rules=conditional_rules(grid[2],10,16,rank,rankstats)
        for representation in ('unconditioned_V','rank_conditioned_W'):
            stats=defaultdict(float);clock=time.process_time();contributions={}
            for j in labels:
                if representation=='unconditioned_V':
                    contributions[j]=integrate_backwards(model,j,grid,cache[f'j{j}'],16,stats,method='cubic')[0]
                else:
                    W=cache[f'j{j}']/q[None,None,:,:]
                    contributions[j]=transport(model,j,grid,W,rules,stats,method='cubic')[0]
            controlstats=defaultdict(float)
            if representation=='unconditioned_V':
                control=integrate_backwards(model,0,grid,constant,16,controlstats,method='cubic')[0]
            else:
                control=transport(model,0,grid,constant/q[None,None,:,:],rules,controlstats,method='cubic')[0]
            result['rows'].append({'grid':n,'representation':representation,'method':'cubic',
                'accuracy':sum(contributions.values()),'contributions':contributions,
                'work':dict(stats),'constant_control':control,'constant_control_error':control-.1,
                'control_work':dict(controlstats),'cpu_seconds':time.process_time()-clock,
                'rank_rule_work':dict(rankstats),'rank_rule_reused':representation=='rank_conditioned_W',
                'constant_terminal_work':dict(cstats),'constant_kernel_reused':representation=='rank_conditioned_W'})
        cache.close()
    result['cpu_seconds_excluding_imports']=time.process_time()-start
    (HERE/'cubic_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
