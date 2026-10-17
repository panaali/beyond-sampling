"""Fixed DEV conditional-law/frozen-map diagnostic; not an estimator."""
import time
START=time.process_time()
import argparse,hashlib,importlib.util,itertools,json,resource
from pathlib import Path
import numpy as np
import torch
from scipy.special import betaincinv,ndtr,ndtri

MOMENT_SOURCE=Path('/workspace/clean-neurons/m16/approaches/crossover_compute/estimator.py')
EXPECTED_MOMENT='68b0b5d7348a55b09a9cb80350a121823923517160bb364cd2b9fcf22bac2a2e'


def estimate(y):
    # Independent samples in each equal-weight chamber; cells are fixed strata.
    return dict(mean=float(y.mean()),se=float(np.sqrt(np.var(y,axis=1,ddof=1).sum()/y.shape[1])/y.shape[0]))


def compare(approx,truth):
    d=approx.astype(float)-truth.astype(float);out=estimate(d)
    bias,se=out['mean'],out['se']
    out.update(approximate=float(approx.mean()),reference=float(truth.mean()),
        disagreement=float(np.mean(approx!=truth)),status='reject_fixed_approximation' if abs(bias)-3*se>.001 else 'advance_diagnostic_only' if abs(bias)+3*se<=.001 else 'inconclusive')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('results.json'));a=p.parse_args()
    assert hashlib.sha256(MOMENT_SOURCE.read_bytes()).hexdigest()==EXPECTED_MOMENT
    spec=importlib.util.spec_from_file_location('pinned_e30_moments',MOMENT_SOURCE);mom=importlib.util.module_from_spec(spec);spec.loader.exec_module(mom)
    start_moment=time.process_time();table=mom.moments(10,2);moment_cpu=time.process_time()-start_moment
    state=torch.load(a.weights,weights_only=True,map_location='cpu')
    wi=state['rnn.weight_ih_l0'].numpy()[:,0].astype(float);wh=state['rnn.weight_hh_l0'].numpy().astype(float);wo=state['linear.weight'].numpy().astype(float)
    base=(wi,wh,wo);sha=hashlib.sha256(a.weights.read_bytes()).hexdigest()
    assert sha=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    rr=np.random.default_rng(28101);gg=np.random.default_rng(28102)
    rank=[];gauss=[];means=[];covs=[];labels=[];maxlabels=[];cellids=[];canonical=[[],[]];canonicalg=[[],[]];cov_min=[]
    count=dict(uniform_variates=0,gaussian_variates=0,beta_inverse_arguments=0,normal_inverse_arguments=0,
        covariance_cholesky=0,model_noise_variates=0,actual_hidden_steps=0,mean_hidden_steps=0,
        rescued_hidden_steps=0,frozen_map_matrix_updates=0,linear_logit_vectors=0,actual_logit_vectors=0,
        rescued_logit_vectors=0,analytic_gaussian_tail_arguments=0)
    for st,(mean,cov) in enumerate(table):
        chol=np.linalg.cholesky(cov);count['covariance_cholesky']+=1;cov_min.append(float(np.linalg.eigvalsh(cov).min()))
        for m,s in itertools.permutations(range(10),2):
            u=rr.random((128,10));q=betaincinv(9,2,(st+u[:,0])/2)
            second=ndtri(q);maximum=ndtri(q+(1-q)*u[:,1]);lower=ndtri(q[:,None]*u[:,2:])
            xx=np.column_stack([maximum,second,lower]);canonical[st].append(xx)
            assert np.all(np.argsort(xx,axis=1)[:,-2:]==np.array([1,0]))
            cdfq=10*q**9-9*q**10
            assert np.all((cdfq>=st/2-1e-12)&(cdfq<=(st+1)/2+1e-12))
            yy=mean+gg.standard_normal((128,10))@chol.T
            canonicalg[st].append(yy)
            order=[m,s]+[i for i in range(10) if i not in (m,s)];inv=np.argsort(order)
            rank.append(xx[:,inv]);gauss.append(yy[:,inv]);means.append(mean[inv]);covs.append(cov[np.ix_(inv,inv)])
            labels.append(s);maxlabels.append(m);cellids.append((st,m,s))
            count['uniform_variates']+=1280;count['gaussian_variates']+=1280
            count['beta_inverse_arguments']+=128;count['normal_inverse_arguments']+=1280
    rank,gauss,means,covs=np.array(rank),np.array(gauss),np.array(means),np.array(covs)
    labels,maxlabels=np.array(labels),np.array(maxlabels);C,N,T=rank.shape;rows=np.arange(C)
    assert (C,N,T)==(180,128,10) and np.isfinite(rank).all() and np.isfinite(gauss).all()
    truth_order=np.argsort(rank,axis=2)
    assert np.all(truth_order[:,:,-1]==maxlabels[:,None]) and np.all(truth_order[:,:,-2]==labels[:,None])
    gauss_order=np.argsort(gauss,axis=2)
    leakage=dict(top_two_chamber_violation=float(np.mean((gauss_order[:,:,-1]!=maxlabels[:,None])|(gauss_order[:,:,-2]!=labels[:,None]))),
                 second_label_violation=float(np.mean(gauss_order[:,:,-2]!=labels[:,None])))
    moment_controls=[]
    for st in range(2):
        xx=np.concatenate(canonical[st]);yy=np.concatenate(canonicalg[st]);mean,cov=table[st]
        moment_controls.append(dict(stratum=st,n=len(xx),max_mean_error=float(np.max(abs(xx.mean(axis=0)-mean))),
            max_covariance_error=float(np.max(abs(np.cov(xx,rowvar=False,ddof=1)-cov))),min_cov_eigenvalue=cov_min[st],
            gaussian_max_mean_error=float(np.max(abs(yy.mean(axis=0)-mean))),
            gaussian_max_covariance_error=float(np.max(abs(np.cov(yy,rowvar=False,ddof=1)-cov)))))
    zero=np.zeros((C,N,10));zero_correct=zero.argmax(axis=2)==labels[:,None]
    assert abs(float(zero_correct.mean())-.1)<1e-15
    reports=[]
    for sigma in [.0025,.02]:
        noise=np.random.default_rng(1);Wi,Wh,Wo=tuple(w+sigma*np.sqrt((w*w).mean())*noise.standard_normal(w.shape) for w in base)
        count['model_noise_variates']+=sum(w.size for w in base)
        h=np.zeros((C,16));J=np.zeros((C,16,10));J8=None
        for t in range(10):
            pre=h@Wh.T+means[:,t,None]*Wi;gate=pre>0
            J=np.einsum('ij,cjk->cik',Wh,J);J[:,:,t]+=Wi;J*=gate[:,:,None]
            h=np.maximum(pre,0)
            if t==7:J8=J.copy()
        count['mean_hidden_steps']+=C*10;count['frozen_map_matrix_updates']+=C*10
        G=np.einsum('ij,cjk->cik',Wo,J);mean_logits=h@Wo.T
        map_error=float(np.max(abs(np.einsum('cij,cj->ci',G,means)-mean_logits)))
        logr=np.einsum('cij,cnj->cni',G,rank);logg=np.einsum('cij,cnj->cni',G,gauss)
        count['linear_logit_vectors']+=2*C*N

        def forward(x):
            flat=x.reshape(-1,10);hh=np.zeros((len(flat),16))
            for t in range(10):hh=np.maximum(hh@Wh.T+flat[:,t,None]*Wi,0)
            count['actual_hidden_steps']+=len(flat)*10;count['actual_logit_vectors']+=len(flat)
            return (hh@Wo.T).reshape(C,N,10)

        actualr=forward(rank);actualg=forward(gauss)
        prefix=np.einsum('cij,cnj->cni',J8,rank);negative_prefix=float(np.mean(np.any(prefix<0,axis=2)))
        hr=prefix.reshape(-1,16);flat=rank.reshape(-1,10)
        for t in [8,9]:hr=np.maximum(hr@Wh.T+flat[:,t,None]*Wi,0)
        rescued=(hr@Wo.T).reshape(C,N,10)
        count['rescued_hidden_steps']+=C*N*2;count['rescued_logit_vectors']+=C*N
        laws=dict(rank_linear=logr,gaussian_linear=logg,rank_network=actualr,gaussian_network=actualg,rank_final2_rescue=rescued)
        correct={k:v.argmax(axis=2)==labels[:,None] for k,v in laws.items()}
        marginals={}
        for name,logits in laws.items():
            target=np.take_along_axis(logits,np.broadcast_to(labels[:,None,None],(C,N,1)),axis=2)
            failure=(logits>target)|((logits==target)&(np.arange(10)[None,None,:]<labels[:,None,None]))
            assert np.array_equal(~failure.any(axis=2),correct[name])
            marginals[name]=failure.mean(axis=1)
        margins=mean_logits[rows,labels,None]-mean_logits
        grad=G[rows,labels,None,:]-G
        var=np.einsum('cij,cjk,cik->ci',grad,covs,grad);sd=np.sqrt(np.maximum(var,0))
        tails=ndtr(-margins/np.maximum(sd,1e-150));zero=sd<1e-14
        deterministic=(margins<0)|((margins==0)&(np.arange(10)[None]<labels[:,None]))
        tails=np.where(zero,deterministic.astype(float),tails);tails[rows,labels]=0
        count['analytic_gaussian_tail_arguments']+=C*10
        analytic_lower=float(1-np.minimum(tails.sum(axis=1),1).mean());analytic_upper=float(1-tails.max(axis=1).mean())
        pinned=mom.evaluate((Wi,Wh,Wo),mom.cost_for(16,10,2)+1e-8)
        # Pinned evaluation builds another mean/Jacobian map; count its work.
        count['mean_hidden_steps']+=C*10;count['frozen_map_matrix_updates']+=C*10
        count['analytic_gaussian_tail_arguments']+=C*10
        assert abs(analytic_lower-pinned['estimate'])<1e-11
        effects=dict(rank_law_effect_frozen=compare(correct['rank_linear'],correct['gaussian_linear']),
            gate_effect_rank=compare(correct['rank_network'],correct['rank_linear']),
            gate_effect_gaussian=compare(correct['gaussian_network'],correct['gaussian_linear']),
            rank_law_effect_network=compare(correct['rank_network'],correct['gaussian_network']),
            rank_linear_bias=compare(correct['rank_linear'],correct['rank_network']),
            gaussian_linear_bias=compare(correct['gaussian_linear'],correct['rank_network']),
            final2_rescue_bias=compare(correct['rank_final2_rescue'],correct['rank_network']))
        interaction=(correct['rank_network'].astype(float)-correct['rank_linear'])-(correct['gaussian_network'].astype(float)-correct['gaussian_linear'])
        tail_changes={}
        for first,last in [('gaussian_linear','rank_linear'),('rank_linear','rank_network'),('rank_final2_rescue','rank_network')]:
            diff=marginals[last]-marginals[first]
            tail_changes[first+'_to_'+last]=dict(mean_abs_competitor_change=float(np.sum(abs(diff))/(C*9)),
                max_abs_cell_competitor_change=float(np.max(abs(diff))),mean_signed_competitor_change=float(np.sum(diff)/(C*9)))
        reports.append(dict(sigma=sigma,noise_seed=1,copy=0,model_array_sha256=hashlib.sha256(b''.join(w.tobytes() for w in (Wi,Wh,Wo))).hexdigest(),
            mean_map_max_abs_error=map_error,law_accuracy={name:estimate(ok.astype(float)) for name,ok in correct.items()},
            effects=effects,law_gate_interaction=estimate(interaction),tail_changes=tail_changes,
            analytic_gaussian_sum_estimate=analytic_lower,analytic_gaussian_union_ceiling=analytic_upper,
            gaussian_marginal_sampling_mean_abs_error=float(np.mean(abs(marginals['gaussian_linear']-tails))),
            rank_frozen_prefix_negative_fraction=negative_prefix,
            empirical_marginals={k:v.tolist() for k,v in marginals.items()}))
    out=dict(checkpoint_sha256=sha,moment_source_sha256=EXPECTED_MOMENT,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells=cellids,cell_n=N,rank_gaussian_input_counts=[C*N,C*N],input_rank_controls_pass=True,
        gaussian_rank_leakage=leakage,moment_controls=moment_controls,constant_zero_accuracy=.1,
        results=reports,counts=count,model_free_moment_cpu_seconds=moment_cpu,
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='DEV conditional Monte Carlo diagnostic; Gaussian chamber labels are pseudo-labels outside rank chamber. Not a deductive estimator or MSE/cost benchmark.')
    assert out['cpu_seconds_including_imports']<20
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['gaussian_rank_leakage','moment_controls','counts','cpu_seconds_including_imports','max_rss_kib']},indent=2))
    for report in reports:print(json.dumps({k:v for k,v in report.items() if k!='empirical_marginals'},indent=2))


if __name__=='__main__':main()
