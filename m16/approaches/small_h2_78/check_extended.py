import json,time,resource
from pathlib import Path
import numpy as np
from partial import partial


def main():
    start=time.process_time();import torch
    p=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
    model=tuple(p[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model=(model[0].reshape(-1),model[1],model[2]);rows=[]
    for order in (2,4):
        r=partial(model,max_suffix=6,order=order,max_cpu=45);r['base_order']=order;rows.append(r)
        print(json.dumps({'base_order':order,'correct_mass':r['partial_correct_mass'],
                          'unresolved':r['unresolved_event_mass'],'stopped':r['stopped_due_cpu_budget'],
                          'cpu':r['cpu_seconds']}),flush=True)
        Path(__file__).with_name('extended_results.json').write_text(json.dumps({'runs':rows},indent=2)+'\n')
    rng=np.random.default_rng(49006);N=65536;x=rng.standard_normal((N,8));h=np.zeros((N,2))
    for t in range(8):h=np.maximum(h@model[1].T+x[:,t,None]*model[0],0)
    correct=np.argmax(h@model[2].T,axis=1)==np.argsort(x,axis=1)[:,-2];ends=np.full(N,-1)
    for t in range(1,8):ends[(x[:,t-1]<0)&(x[:,t]<0)]=t
    validation=[]
    for component in rows[-1]['rows']:
        length=component['suffix_length'];v=correct&(ends==7-length)
        q=float(v.mean());se=float(np.sqrt(q*(1-q)/N));pred=component['contribution']
        validation.append({'suffix_length':length,'outer_approx':pred,'mc':q,'se':se,'z':(q-pred)/se})
    out={'runs':rows,'validation':validation,'no_reset_diagnostic_correct_mass':float(np.mean(correct&(ends<0))),
         'compute':{'cpu_seconds':time.process_time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'validation_inputs':N,'random_inputs_for_estimation':0}}
    Path(__file__).with_name('extended_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'validation':validation,'compute':out['compute']},indent=2))
if __name__=='__main__':main()
