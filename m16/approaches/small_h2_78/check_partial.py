import json,time
from pathlib import Path
import numpy as np
from partial import partial

def main():
    start=time.process_time()
    import torch
    p=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
    model=tuple(p[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model=(model[0].reshape(-1),model[1],model[2])
    rows=[]
    for order in (8,16,32):
        row=partial(model,max_suffix=3,order=order);row['order']=order;rows.append(row)
    # Independent full inputs diagnose each disjoint branch, never used by estimator.
    rng=np.random.default_rng(49004);N=65536;x=rng.standard_normal((N,8));h=np.zeros((N,2))
    for t in range(8):h=np.maximum(h@model[1].T+x[:,t,None]*model[0],0)
    correct=np.argmax(h@model[2].T,axis=1)==np.argsort(x,axis=1)[:,-2]
    ends=np.full(N,-1)
    for t in range(1,8):ends[(x[:,t-1]<0)&(x[:,t]<0)]=t
    validation=[]
    for length in range(4):
        indicator=correct&(ends==7-length);p=float(indicator.mean());se=float(np.sqrt(p*(1-p)/N))
        predicted=rows[-1]['rows'][length]['contribution']
        validation.append({'suffix_length':length,'predicted':predicted,'mc':p,'se':se,'z':(p-predicted)/se})
    assert max(abs(r['z']) for r in validation)<5
    out={'runs':rows,'validation':validation,'compute':{'cpu_seconds':time.process_time()-start,'validation_full_inputs':N,
          'analytic_random_inputs':0,'total_conditional_tail_calls':sum(r['stats']['conditional_tail_calls'] for r in rows)}}
    Path(__file__).with_name('partial_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'results':[{'order':r['order'],'correct_mass':r['partial_correct_mass'],'unresolved':r['unresolved_event_mass']} for r in rows],
                      'validation':validation,'compute':out['compute']},indent=2))
if __name__=='__main__':main()
