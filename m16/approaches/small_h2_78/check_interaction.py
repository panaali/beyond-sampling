"""No-new-kernel replay of the missing angle16 factorial baseline."""
from collections import defaultdict
import json,time
import numpy as np
from curved_transport import (HERE,RankRules,conditional_rules,stage_grids,
    transport,target_probability,nonvanishing_certificate)

started=time.process_time()
data=np.load(HERE/'LOCALPILOT_H2_T7_seed0.npz');model=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))
T=len(model[2]);labels=nonvanishing_certificate(model)['nonzero_state_output_labels']
stats=defaultdict(float);rank=RankRules(64,stats);b=np.linspace(-1,1,5)
rules=conditional_rules(b,T,16,rank,stats);constant=(model[0],model[1],np.zeros_like(model[2]))
q=np.zeros((5,2));qwork=defaultdict(float)
for ib,bv in enumerate(b):
    for count in (0,1):
        q[ib,count],work=target_probability(constant,np.zeros(2),float(bv),count,0);qwork['calls']+=1
        for key,value in work.items():
            if isinstance(value,(int,float)) and key not in ('partition_mass','target_index','prefix_dimension'):qwork[key]+=value
estimates={};provenance={}
for j in labels:
    path=HERE/f'T7_curved_h5_b17_j{j}.npz';raw=np.load(path)['raw'][:,:,::4,:]
    provenance[j]=json.loads(path.with_suffix('.json').read_text())
    grids,boxes=stage_grids(model,j,5,5);W=raw/q[None,None,:,:]
    assert W.min()>=-1e-10 and W.max()<=1+1e-8
    estimates[j],_=transport(model,j,grids,boxes,W,rules,stats)
cg,cb=stage_grids(model,0,5,5);ones=np.ones((5,5,5,2))
control,_=transport(constant,0,cg,cb,ones,rules,stats);assert abs(control-1/T)<1e-12
result={'hidden_nodes':5,'target_nodes':5,'angles':16,'rank_nodes':64,'accuracy':sum(estimates.values()),
    'contributions':estimates,'constant_W_error':control-1/T,'new_neural_terminal_calls':0,
    'rank_tail_work':dict(qwork),'transport_work':dict(stats),'source_table_provenance':provenance,
    'cpu_seconds_excluding_imports':time.process_time()-started,'model_input_samples':0}
(HERE/'factorial_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('accuracy','constant_W_error','cpu_seconds_excluding_imports')}))
