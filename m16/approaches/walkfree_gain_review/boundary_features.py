"""Frozen true-tie-facet feature rescue; no gain-table access."""
import time
START=time.process_time()
import hashlib,importlib.util,itertools,json,resource
from pathlib import Path
import numpy as np
import torch

SOURCE=Path('/workspace/clean-neurons/m16/approaches/crossover_compute/estimator.py')
HASH='68b0b5d7348a55b09a9cb80350a121823923517160bb364cd2b9fcf22bac2a2e'
COUNTS=dict(checkpoint_loads=0,feature_calls=0,boundary_probe_inputs=0,hidden_steps=0,
            jacobian_matrix_updates=0,margin_gradients=0,model_free_moment_tables=0)


def feature(model,mean):
    wi,wh,wo=model;H=len(wi);T=len(mean);xs=[];labels=[];competitors=[];normals=[];maxs=[]
    for m,s in itertools.permutations(range(T),2):
        order=[m,s]+[i for i in range(T) if i not in (m,s)];mu=mean[np.argsort(order)]
        for k in range(T):
            if k==s:continue
            normal=np.zeros(T)
            if k==m:normal[m],normal[s]=1,-1
            else:normal[s],normal[k]=1,-1
            face=mu-normal*(normal@mu)/2
            probe=face+2**-20*np.linalg.norm(face)*normal
            assert np.argsort(probe)[-2:].tolist()==[s,m]
            xs.append(probe);labels.append(s);competitors.append(k);normals.append(normal);maxs.append(m)
    x=np.array(xs);normals=np.array(normals);labels=np.array(labels);competitors=np.array(competitors);n=len(x)
    h=np.zeros((n,H));J=np.zeros((n,H,T))
    for t in range(T):
        pre=h@wh.T+x[:,t,None]*wi;gate=pre>0
        J=np.einsum('ij,njk->nik',wh,J);J[:,:,t]+=wi;J*=gate[:,:,None]
        h=np.maximum(pre,0)
    G=np.einsum('ij,njk->nik',wo,J);rows=np.arange(n);grad=G[rows,labels]-G[rows,competitors]
    norm=np.linalg.norm(grad,axis=1);cos=np.divide(np.sum(grad*normals,axis=1),norm*np.sqrt(2),out=np.zeros(n),where=norm>0)
    assert np.max(abs(cos))<1+1e-12
    error=float(np.max(abs(np.einsum('nij,nj->ni',G,x)-h@wo.T)))
    COUNTS['feature_calls']+=1;COUNTS['boundary_probe_inputs']+=n;COUNTS['hidden_steps']+=n*T
    COUNTS['jacobian_matrix_updates']+=n*T;COUNTS['margin_gradients']+=n
    return dict(boundary_mean_alignment=float(cos.mean()),boundary_p10_alignment=float(np.quantile(cos,.1)),
        zero_margin_gradients=int(np.sum(norm==0)),max_mean_map_error=error,
        probe_count=n,epsilon_relative=2**-20)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==HASH
    spec=importlib.util.spec_from_file_location('pinned_rank_moments',SOURCE);mom=importlib.util.module_from_spec(spec);spec.loader.exec_module(mom)
    manifest_path=Path(__file__).with_name('checkpoint_manifest.json');manifest=json.loads(manifest_path.read_text());assert manifest['available']==90
    then=time.process_time();tables={T:mom.moments(T,1)[0][0] for T in [3,4,5,6,8,10]};moment_cpu=time.process_time()-then
    COUNTS['model_free_moment_tables']=6;rows=[];checks=[]
    for i,item in enumerate(manifest['models']):
        path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        state=torch.load(path,weights_only=True,map_location='cpu');COUNTS['checkpoint_loads']+=1
        wi=state['rnn.weight_ih_l0'].numpy()[:,0].astype(float);wh=state['rnn.weight_hh_l0'].numpy().astype(float);wo=state['linear.weight'].numpy().astype(float)
        H,T,seed=item['H'],item['T'],item['seed'];mean=tables[T];base=feature((wi,wh,wo),mean)
        rows.append(dict(H=H,T=T,seed=seed,checkpoint_sha256=item['sha256'],**base))
        if (H,T,seed) in [(8,3,0),(16,6,0),(32,10,0)]:
            d=np.logspace(-1,1,H);perm=np.arange(H)[::-1]
            variants={'hidden_positive_coordinate_change':(d*wi,d[:,None]*wh/d[None,:],wo/d[None,:]),
                'hidden_permutation':(wi[perm],wh[np.ix_(perm,perm)],wo[:,perm]),'output_scale_1e-8':(wi,wh,wo*1e-8)}
            for name,model in variants.items():
                val=feature(model,mean);errors={k:abs(val[k]-base[k]) for k in ['boundary_mean_alignment','boundary_p10_alignment']}
                assert max(errors.values())<1e-10
                checks.append(dict(H=H,T=T,seed=seed,control=name,absolute_errors=errors))
        if (i+1)%30==0:print(f'Boundary feature {i+1}/90 complete; gain table not read',flush=True)
        if time.process_time()-START>18:raise RuntimeError('Frozen review CPU deadline reached')
    out=dict(models=rows,invariance_checks=checks,counts=COUNTS,moment_source_sha256=HASH,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_free_moment_cpu_seconds=moment_cpu,cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note='Post-initial-result frozen boundary representation rescue; feature file saved before its gain association; no face walks.')
    Path(__file__).with_name('boundary_features.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['models','invariance_checks']},indent=2))


if __name__=='__main__':main()
