"""Frozen T45 features; deliberately never opens gain or face-walk data."""
import time
START=time.process_time()
import hashlib,importlib.util,itertools,json,resource
from pathlib import Path
import numpy as np
import torch

SOURCE=Path('/workspace/clean-neurons/m16/approaches/crossover_compute/estimator.py')
HASH='68b0b5d7348a55b09a9cb80350a121823923517160bb364cd2b9fcf22bac2a2e'
COUNTS=dict(checkpoint_loads=0,feature_calls=0,canonical_cells=0,hidden_steps=0,
            jacobian_matrix_updates=0,covariance_quadratic_forms=0,margin_gradients=0,
            model_free_moment_tables=0)


def feature(model,mean,cov):
    wi,wh,wo=model;H=len(wi);T=len(mean)
    pairs=list(itertools.permutations(range(T),2));C=len(pairs)
    orders=np.array([[m,s]+[i for i in range(T) if i not in (m,s)] for m,s in pairs]);inverse=np.argsort(orders,axis=1)
    means=mean[inverse];covs=np.array([cov[np.ix_(v,v)] for v in inverse]);m,s=np.array(pairs).T
    h=np.zeros((C,H));J=np.zeros((C,H,T));uncertainty=[];max_map_error=0.;minimum_raw_variance=0.;negative_variances=0;zero_forms=0
    for t in range(T):
        pre=h@wh.T+means[:,t,None]*wi
        Jpre=np.einsum('ij,cjk->cik',wh,J);Jpre[:,:,t]+=wi
        v=np.einsum('cij,cjk,cik->ci',Jpre,covs,Jpre)
        minimum_raw_variance=min(minimum_raw_variance,float(v.min()));negative_variances+=int(np.sum(v<0));v=np.maximum(v,0)
        den=v+pre*pre;proxy=np.divide(v,den,out=np.zeros_like(v),where=den>0)
        zero_forms+=int(np.sum(np.all(Jpre==0,axis=2)));uncertainty.append(proxy)
        max_map_error=max(max_map_error,float(np.max(abs(np.einsum('cij,cj->ci',Jpre,means)-pre))))
        gates=pre>0;h=np.maximum(pre,0);J=Jpre*gates[:,:,None]
    G=np.einsum('ij,cjk->cik',wo,J);rows=np.arange(C);alignment=[];zeros=0
    for k in range(T):
        keep=k!=s;ids=rows[keep]
        grad=G[ids,s[keep]]-G[ids,k]
        norm=np.linalg.norm(grad,axis=1);dot=np.where(k==m[keep],grad[np.arange(len(ids)),k]-grad[np.arange(len(ids)),s[keep]],
                                                  grad[np.arange(len(ids)),s[keep]]-grad[np.arange(len(ids)),k])
        cosine=np.divide(dot,norm*np.sqrt(2),out=np.zeros_like(dot),where=norm>0)
        assert np.max(abs(cosine))<=1+1e-12
        zeros+=int(np.sum(norm==0));alignment.extend(cosine.tolist())
    COUNTS['feature_calls']+=1;COUNTS['canonical_cells']+=C;COUNTS['hidden_steps']+=C*T
    COUNTS['jacobian_matrix_updates']+=C*T;COUNTS['covariance_quadratic_forms']+=C*T*H;COUNTS['margin_gradients']+=C*(T-1)
    return dict(gate_uncertainty=float(np.mean(uncertainty)),boundary_alignment=float(np.mean(alignment)),
        zero_preactivation_forms=zero_forms,zero_margin_gradients=zeros,
        max_mean_map_error=max_map_error,min_raw_covariance_quadratic=minimum_raw_variance,
        negative_covariance_quadratic_count=negative_variances)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==HASH
    spec=importlib.util.spec_from_file_location('pinned_rank_moments',SOURCE);mom=importlib.util.module_from_spec(spec);spec.loader.exec_module(mom)
    manifest_path=Path(__file__).with_name('checkpoint_manifest.json');manifest=json.loads(manifest_path.read_text())
    assert manifest['available']==90
    start_moment=time.process_time();tables={T:mom.moments(T,1)[0] for T in [3,4,5,6,8,10]};moment_cpu=time.process_time()-start_moment
    COUNTS['model_free_moment_tables']=len(tables)
    rows=[];checks=[];started=time.process_time()
    for i,item in enumerate(manifest['models']):
        path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        state=torch.load(path,weights_only=True,map_location='cpu');COUNTS['checkpoint_loads']+=1
        wi=state['rnn.weight_ih_l0'].numpy()[:,0].astype(float);wh=state['rnn.weight_hh_l0'].numpy().astype(float);wo=state['linear.weight'].numpy().astype(float)
        H,T,seed=item['H'],item['T'],item['seed'];assert wi.shape==(H,) and wh.shape==(H,H) and wo.shape==(T,H)
        mean,cov=tables[T];base=feature((wi,wh,wo),mean,cov)
        rows.append(dict(H=H,T=T,seed=seed,checkpoint_sha256=item['sha256'],**base))
        if (H,T,seed) in [(8,3,0),(16,6,0),(32,10,0)]:
            d=np.logspace(-1,1,H);perm=np.arange(H)[::-1]
            transformed={
                'hidden_positive_coordinate_change':(d*wi,d[:,None]*wh/d[None,:],wo/d[None,:]),
                'hidden_permutation':(wi[perm],wh[np.ix_(perm,perm)],wo[:,perm]),
                'output_scale_1e-8':(wi,wh,wo*1e-8)}
            for label,model in transformed.items():
                actual=feature(model,mean,cov)
                errors={k:abs(actual[k]-base[k]) for k in ['gate_uncertainty','boundary_alignment']}
                assert max(errors.values())<1e-10
                checks.append(dict(H=H,T=T,seed=seed,control=label,absolute_errors=errors))
        if (i+1)%30==0:print(f'Computed {i+1}/90 frozen features; no gain data read',flush=True)
        if time.process_time()-START>100:raise RuntimeError('Frozen CPU cap reached')
    result=dict(models=rows,invariance_checks=checks,counts=COUNTS,
        moment_source_sha256=HASH,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_free_moment_cpu_seconds=moment_cpu,
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note='Features saved independently of gain-table access; no F values or walker calls.')
    Path(__file__).with_name('features.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['models','invariance_checks']},indent=2))


if __name__=='__main__':main()
