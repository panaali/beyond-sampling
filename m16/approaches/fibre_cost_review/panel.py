from pathlib import Path
import hashlib,json,time
import numpy as np
from instrument import evaluate
from sample_count import count

def main():
    start=time.process_time();zero=(np.zeros(4),np.zeros((4,4)),np.zeros((3,4)))
    positive=(np.ones(4),np.ones((4,4))/8,np.array([[1.,1.,1.,1.],[2.,2.,2.,2.],[3.,3.,3.,3.]]))
    models=[('zero',zero,None),('positive',positive,None)]
    import torch
    for k,seed in ((30,0),(30,1),(30,4),(24,4)):
        path=Path('/workspace/work/t114_checkpoints')/f'2nd_argmax_4_3_{2**k}_{seed}.pth';sd=torch.load(path,map_location='cpu',weights_only=True)
        model=(sd['rnn.weight_ih_l0'][:,0].double().numpy().copy(),sd['rnn.weight_hh_l0'].double().numpy().copy(),sd['linear.weight'].double().numpy().copy());digest=hashlib.sha256(path.read_bytes()).hexdigest();models.append((f'k{k}_s{seed}_clean',model,digest))
        rng=np.random.default_rng(124);noisy=tuple(a+.05*rng.standard_normal(a.shape) for a in model);models.append((f'k{k}_s{seed}_abs05_n124_0',noisy,digest))
    rows=[]
    for name,model,digest in models:
        stamp=time.process_time();baseline=evaluate(model,False);audit=evaluate(model,True)
        parity={key:baseline[key]==audit[key] for key in ('value_hex','n','x','segments','traces')};assert all(parity.values()),(name,parity)
        if name=='zero':assert abs(audit['value']-1/3)<1e-14
        sample=count(model);row={'name':name,'checkpoint_sha256':digest,'baseline':baseline,'audit':audit,'parity':parity,'sampling_scalar':sample,'historical_sample_equivalents':audit['historical_hybrid']/72,'partial_adjusted_sample_equivalents':audit['partial_scalar_adjustment']/sample['total'],'cpu_seconds':time.process_time()-stamp}
        row['partial_vs_historical_budget_ratio']=row['partial_adjusted_sample_equivalents']/row['historical_sample_equivalents'];rows.append(row)
        print(json.dumps({'name':name,'value':audit['value'],'historical':audit['historical_hybrid'],'matrix_corrected_partial':audit['partial_scalar_adjustment'],'budget_ratio_partial':row['partial_vs_historical_budget_ratio'],'cpu_seconds':row['cpu_seconds']}),flush=True)
    out={'rows':rows,'cpu_seconds':time.process_time()-start,'complete_count':False,'scope':'frozen ten cases, dense/mask correction and common-unit sampler; remaining scalar audit open'}
    Path(__file__).with_name('panel_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
