"""Frozen post-review support repair on a fresh validation stream; no refit."""
import hashlib,json,time,resource
from pathlib import Path
import numpy as np
from probe import reference_model,states,recurrence,score
HERE=Path(__file__).parent


def clipped(model,b,x,r):
    wi,w,wo=model;h=np.zeros((len(x),len(wi)));negative=[];sizes=[]
    for t in range(x.shape[1]):
        bar=r[:,t]@b.T;z=bar@w.T+x[:,t,None]*wi
        before=np.maximum(z,0)+(z>0)*((h-bar)@w.T)
        negative.append(float(np.mean(before<0)));sizes.append(float(np.sqrt(np.mean(np.minimum(before,0)**2))))
        h=np.maximum(before,0)
    return h@wo.T,negative,sizes


def run():
    start=time.process_time();original_path=HERE/'results.json';old=json.loads(original_path.read_text())
    ref=reference_model();test=np.stack((np.linspace(-1,1,10),np.zeros(10),-np.ones(10)))
    rr=states(ref,test);p,_,_=clipped(ref,np.eye(22),test,rr);assert np.max(abs(p-rr[:,-1]@ref[2].T))<1e-12
    import torch
    checkpoint=HERE.parent/'face_surplus_review/data/H16_T10_original.pth';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    base=[state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];base[0]=base[0][:,0]
    x=np.random.default_rng(370203).standard_normal((32768,10));truth=np.argsort(x,axis=1)[:,-2]
    r=np.concatenate([states(ref,x[i:i+4096]) for i in range(0,len(x),4096)])
    out=dict(protocol='dfe9504',calibration_artifact_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),validation_seed=370203,validation_n=len(x),models=[])
    for saved in old['models']:
        sigma=saved['sigma'];rng=np.random.default_rng(1);model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base);b=np.array(saved['alignment'])
        values={key:[] for key in ('actual','linear','clipped')};negative=[];sizes=[]
        for i in range(0,len(x),4096):
            original,_=recurrence(model,b,x[i:i+4096],r[i:i+4096]);logits,n,s=clipped(model,b,x[i:i+4096],r[i:i+4096])
            values['actual'].append(original['actual']);values['linear'].append(original['linear']);values['clipped'].append(logits);negative.append(n);sizes.append(s)
        values={key:np.concatenate(v) for key,v in values.items()}
        row=dict(sigma=sigma,rules={key:score(values['actual'],values[key],truth) for key in ('linear','clipped')},
            step_preclip_negative_fraction=np.mean(negative,axis=0).tolist(),step_preclip_negative_rms=np.sqrt(np.mean(np.array(sizes)**2,axis=0)).tolist())
        out['models'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/'support_rescue.json').write_text(json.dumps(out,indent=2)+'\n')
        print(sigma,[(key,value['paired_bias'],value['paired_se'],value['diagnostic_gate']) for key,value in row['rules'].items()],flush=True)


if __name__=='__main__':run()
