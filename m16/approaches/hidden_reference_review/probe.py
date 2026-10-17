"""Calibrated hidden-reference ceiling; sampled diagnostic, never an estimator."""
import hashlib,json,math,resource,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent


def reference_model():
    from alg_zoo.handcrafted import handcrafted_2nd_argmax_10
    net=handcrafted_2nd_argmax_10();s=net.state_dict()
    w=[s[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')]
    return w[0][:,0],w[1],w[2]


def states(model,x):
    wi,w,_=model;h=np.zeros((len(x),x.shape[1]+1,len(wi)))
    for t in range(x.shape[1]):h[:,t+1]=np.maximum(h[:,t]@w.T+x[:,t,None]*wi,0)
    return h


def closed_reference(x):
    p=np.maximum(x,0);answer=p.copy()
    for j in range(x.shape[1]):answer[:,j]+=2*np.max(np.delete(p,j,axis=1),axis=1)
    return answer


def recurrence(model,b,x,r):
    wi,w,wo=model;n,tmax=x.shape;h=np.zeros((n,len(wi)));linear=h.copy();q=h.copy()
    stats={'identity_max':0.,'kappa_min':0.,'kappa_upper_violation':0.,'step_crossing_fraction':[],
        'step_kappa_rms':[],'hidden_difference_square_sum':0.,'hidden_square_sum':0.,'aligned_difference_square_sum':0.,'hidden_entries':0}
    for t in range(tmax):
        bar=r[:,t]@b.T;forcing=x[:,t,None]*wi;z=bar@w.T+forcing;d=(h-bar)@w.T;gate=z>0
        reluz=np.maximum(z,0);kappa=np.maximum(z+d,0)-reluz-gate*d
        if t==tmax-1:rescued=np.maximum(linear@w.T+forcing,0)
        h=np.maximum(h@w.T+forcing,0)
        linear=reluz+gate*((linear-bar)@w.T)
        q=gate*(q@w.T)+kappa
        stats['identity_max']=max(stats['identity_max'],float(np.max(abs(h-linear-q))))
        stats['kappa_min']=min(stats['kappa_min'],float(kappa.min()))
        upper=abs(d)*(abs(z)<=abs(d))
        stats['kappa_upper_violation']=max(stats['kappa_upper_violation'],float(np.max(kappa-upper)))
        stats['step_crossing_fraction'].append(float(np.mean(gate!=(z+d>0))))
        stats['step_kappa_rms'].append(float(np.sqrt(np.mean(kappa*kappa))))
        stats['hidden_difference_square_sum']+=float(np.sum((h-linear)**2));stats['hidden_square_sum']+=float(np.sum(h*h))
        stats['aligned_difference_square_sum']+=float(np.sum((h-r[:,t+1]@b.T)**2));stats['hidden_entries']+=h.size
    logits={'actual':h@wo.T,'linear':linear@wo.T,'last_exact':rescued@wo.T,'aligned':r[:,-1]@b.T@wo.T}
    stats['output_identity_max']=float(np.max(abs(logits['actual']-logits['linear']-q@wo.T)))
    return logits,stats


def controls():
    ref=reference_model();t=10
    x=np.stack((np.linspace(-1,1,t),-np.ones(t),np.zeros(t),np.arange(t)/8,np.r_[1.,-np.ones(t-1)],np.r_[-1.,np.zeros(t-1)]))
    r=states(ref,x);logits=r[:,-1]@ref[2].T;closed=closed_reference(x)
    assert np.max(abs(logits-closed))<1e-12
    a,s=recurrence(ref,np.eye(22),x,r)
    assert np.max(abs(a['actual']-a['linear']))<1e-12 and max(s['step_kappa_rms'])<1e-12
    zero=(np.array([1.,-2.]),np.zeros((2,2)),np.arange(20).reshape(10,2)/8)
    b=np.arange(44).reshape(2,22)/32
    z,ss=recurrence(zero,b,x,r)
    assert np.array_equal(z['actual'],z['linear'])
    # Scalar exact examples cover both crossing directions and z=0 convention.
    origin=np.array([-2.,-1.,0.,0.,1.,2.]);delta=np.array([3.,.5,-1.,1.,-2.,-.5]);gate=origin>0
    k=np.maximum(origin+delta,0)-np.maximum(origin,0)-gate*delta
    assert np.all(k>=0) and np.all(k<=abs(delta)*(abs(origin)<=abs(delta)))
    return dict(reference_closed_error=float(np.max(abs(logits-closed))),reference_tie_labels=np.argmax(logits,axis=1).tolist(),
        identity_model=s,zero_recurrence=ss,scalar_kappa=k.tolist())


def score(actual,approx,target):
    a=np.argmax(actual,axis=1);b=np.argmax(approx,axis=1);ok=a==target;other=b==target;diff=other.astype(float)-ok
    bias=float(diff.mean());se=float(diff.std(ddof=1)/math.sqrt(len(diff)))
    status='rejected' if abs(bias)-3*se>.001 else 'advance' if abs(bias)+3*se<=.001 else 'inconclusive'
    centered=(approx-actual);centered-=centered.mean(axis=1,keepdims=True)
    def margin(logits):
        z=logits.copy();z[np.arange(len(z)),target]=-np.inf
        return logits[np.arange(len(z)),target]-np.max(z,axis=1)
    gap=margin(approx)-margin(actual)
    return dict(actual_accuracy=float(ok.mean()),approximate_accuracy=float(other.mean()),paired_bias=bias,paired_se=se,
        bias_minus_3se=bias-3*se,bias_plus_3se=bias+3*se,diagnostic_gate=status,label_disagreement=float(np.mean(a!=b)),
        centered_logit_error_rms=float(np.sqrt(np.mean(centered**2))),target_margin_error_rms=float(np.sqrt(np.mean(gap*gap))))


def run():
    started=time.process_time();out=dict(protocol='478f0e2',controls=controls(),models=[])
    import torch
    path=HERE.parent/'face_surplus_review/data/H16_T10_original.pth';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    state=torch.load(path,map_location='cpu',weights_only=True)
    base=[state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];base[0]=base[0][:,0]
    calibration=np.random.default_rng(370201).standard_normal((4096,10));held=np.random.default_rng(370202).standard_normal((32768,10))
    ref=reference_model();rc=states(ref,calibration)
    rh=np.concatenate([states(ref,held[i:i+4096]) for i in range(0,len(held),4096)])
    design=rc[:,1:].reshape(-1,22);target=np.argsort(held,axis=1)[:,-2]
    out.update(checkpoint_sha256=sha,calibration_seed=370201,validation_seed=370202,calibration_n=4096,validation_n=32768,
        exact_reference_forward_sequences=36864,exact_trained_forward_sequences=73728,reference_accuracy=float(np.mean(np.argmax(rh[:,-1]@ref[2].T,axis=1)==target)))
    for sigma in (.0025,.02):
        begin=time.process_time();rng=np.random.default_rng(1);weights=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        hc=states(weights,calibration);fitstart=time.process_time()
        coefficients,residuals,rank,singular=np.linalg.lstsq(design,hc[:,1:].reshape(-1,16),rcond=None);b=coefficients.T
        fitcpu=time.process_time()-fitstart
        fitted=design@coefficients;calmse=float(np.mean((hc[:,1:].reshape(-1,16)-fitted)**2))
        all_logits={key:[] for key in ('actual','linear','last_exact','aligned')};chunks=[]
        for i in range(0,len(held),4096):
            logits,s=recurrence(weights,b,held[i:i+4096],rh[i:i+4096]);chunks.append(s)
            for key,value in logits.items():all_logits[key].append(value)
        values={key:np.concatenate(v) for key,v in all_logits.items()}
        stats={key:max(s[key] for s in chunks) for key in ('identity_max','output_identity_max','kappa_upper_violation')}
        stats['kappa_min']=min(s['kappa_min'] for s in chunks)
        stats['step_crossing_fraction']=np.mean([s['step_crossing_fraction'] for s in chunks],axis=0).tolist()
        stats['step_kappa_rms']=np.sqrt(np.mean(np.array([s['step_kappa_rms'] for s in chunks])**2,axis=0)).tolist()
        entries=sum(s['hidden_entries'] for s in chunks)
        for name in ('hidden_difference','aligned_difference','hidden'):
            stats[name+'_rms']=math.sqrt(sum(s[name+'_square_sum'] for s in chunks)/entries)
        assert stats['identity_max']<1e-8 and stats['output_identity_max']<1e-7 and stats['kappa_min']>-1e-10 and stats['kappa_upper_violation']<1e-10
        row=dict(sigma=sigma,alignment=b.tolist(),fit_rank=int(rank),fit_singular_values=singular.tolist(),fit_cpu=fitcpu,
            calibration_hidden_rms_error=math.sqrt(calmse),heldout_diagnostics=stats,rules={key:score(values['actual'],values[key],target) for key in ('aligned','linear','last_exact')},
            cpu_seconds=time.process_time()-begin)
        out['models'].append(row);out['cpu_seconds']=time.process_time()-started;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
        print('sigma',sigma,'rank',rank,'identity',stats['identity_max'],'rules',[(key,v['paired_bias'],v['paired_se'],v['diagnostic_gate']) for key,v in row['rules'].items()],flush=True)


if __name__=='__main__':run()
