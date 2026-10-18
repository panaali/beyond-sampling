"""Independent fullfit saved contraction; no author imports, fitting, RNN or draws."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,resource,time,traceback
import numpy as np
from scipy.special import betainc
HERE=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_78/review/mechanism_claims_review/rebuild')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bh(a):return hashlib.sha256(a.tobytes()).hexdigest()
def close(a,b,rtol=1e-10,floor=1e-12):assert abs(a-b)<=rtol*max(abs(a),abs(b),floor),(a,b)
def load(meta):
    path=Path(meta['path']);assert sha(path)==meta['sha256']
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(meta['arrays']);arr={k:z[k].copy() for k in z.files}
    for k,v in arr.items():assert meta['arrays'][k]==dict(shape=list(v.shape),dtype=str(v.dtype),bytes_sha256=bh(v))
    return arr
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def valid_state(s,initial):
    assert set(s)==set(initial) and s['bit_generator']=='PCG64' and set(s['state'])=={'state','inc'}
    assert s['state']['inc']==initial['state']['inc'] and type(s['state']['state'])is int and 0<=s['state']['state']<2**128
    assert type(s['has_uint32'])is int and s['has_uint32']in(0,1) and type(s['uinteger'])is int

def metadata(x):
    n=len(x);head=np.argsort(x[:,:9],axis=1)[:,-2:];p2=head[:,0];p1=head[:,1];rows=np.arange(n)
    m1=x[rows,p1];m2=x[rows,p2];last=x[:,9]
    case=np.full(n,2,np.int64);case[last>m2]=1;case[last>m1]=0
    ranked=np.sort(x,axis=1);assert np.all(np.diff(ranked,axis=1)>0)
    true=np.argmax(x==ranked[:,-2,None],axis=1);key=(case*9+p1)*9+p2
    legal=[(c*9+a)*9+b for c in range(3) for a in range(9) for b in range(9) if a!=b];mapping={k:i for i,k in enumerate(legal)}
    slot=np.array([mapping[int(k)] for k in key],dtype=np.int64)
    return dict(case=case,p1=p1,p2=p2,m1=m1,m2=m2,last=last,true=true,key=key,slot=slot,positive_bin=np.minimum((x>0).sum(axis=1),2))
def cp(saved,k,n):
    if n==0:assert saved is None;return 0
    def inverse(p,a,b):
        lo=0.;hi=1.
        for _ in range(80):
            mid=(lo+hi)/2
            if betainc(a,b,mid)<p:lo=mid
            else:hi=mid
        return (lo+hi)/2
    lo=0. if k==0 else inverse(.025,k,n-k+1);hi=1. if k==n else inverse(.975,k+1,n-k)
    assert saved['level']==.95 and saved['method']=='ordinary numerical equal-tail Clopper-Pearson; individual fixed-fit interval'
    assert abs(saved['lower']-lo)<2e-12 and abs(saved['upper']-hi)<2e-12 and lo<=k/n<=hi
    return int(k>0)+int(k<n)
def hidden_metrics(saved,h,pred):
    n,K=h.shape;sse=[];sst=[];r2=[]
    for j in range(K):
        truth=h[:,j];mean=math.fsum(map(float,truth))/n
        e=math.fsum(float(v)*float(v) for v in pred[:,j]-truth)
        variance=math.fsum((float(v)-mean)**2 for v in truth)
        sse.append(e);sst.append(variance);r2.append(None if variance==0 else 1-e/variance)
    assert saved['n']==n and saved['zero_variance_columns']==[j for j,v in enumerate(r2) if v is None]
    for field,vals in [('sse',sse),('sst',sst),('mse',[v/n for v in sse])]:
        assert len(saved[field])==K
        for a,b in zip(saved[field],vals,strict=True):close(a,b)
    for a,b in zip(saved['r2'],r2,strict=True):
        if b is None:assert a is None
        else:close(a,b)
    ordered=sorted(v for v in r2 if v is not None)
    med=None if not ordered else (ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2
    if med is None:assert saved['median_r2'] is None
    else:close(saved['median_r2'],med)
    return dict(median_r2=med,maximum_neuron_mse=max(sse)/n)
def classify(saved,x,s,pred,network,heldout):
    n=len(x);correct=int((pred==s['true']).sum());agree=int((pred==network).sum());ranks=1+(x>x[np.arange(n),pred,None]).sum(axis=1)
    keys=(s['positive_bin']*3+s['case'])*10+ranks-1;table=np.bincount(keys,minlength=90).reshape(3,3,10)
    assert saved['n']==n and saved['task_correct']==correct and saved['network_agree']==agree
    assert saved['task_accuracy']==correct/n and saved['network_agreement']==agree/n
    assert saved['cross_tab']==dict(n=n,correct=correct,table=table.tolist()) and int(table[:,:,1].sum())==correct
    beta=0
    if heldout:beta+=cp(saved['task_interval'],correct,n)+cp(saved['agreement_interval'],agree,n)
    else:assert saved['task_interval'] is None and saved['agreement_interval'] is None
    scope='fixed model and fixed fitted coefficients; individual, not simultaneous' if heldout else 'training data: no fixed-fit binomial interval'
    assert saved['interval_scope']==scope
    return table,beta

def check():
    manifest=read(HERE/'pins.json')
    for p,h in manifest['pins'].items():assert sha(p)==h,p
    p=read(A/'FULLFIT_PROTOCOL.json');gate=read(A/'fullfit_gate.json');d=read(A/'fullfit/results.json');pilot=read(A/'pilot/results.json')
    assert d['complete'] and gate['complete'] and d['protocol_sha256']==gate['protocol_sha256']==sha(A/'FULLFIT_PROTOCOL.json') and d['gate_sha256']==sha(A/'fullfit_gate.json')
    for path,h in {**p['pins'],**gate['pins']}.items():assert sha(path)==h,path
    assert d['scope']==p['scope'] and d['runtime_versions']==dict(numpy=p['numpy_version'],scipy=p['scipy_version'],python=p['python_version'])
    assert p['seeds']==read(A/'PILOT_PROTOCOL.json')['seeds'] and len(set(p['seeds'].values()))==3
    coef=load(d['lookup_coefficients'])['coefficients'];q=load(d['separable_coefficients'])['coefficients']
    assert coef.shape==(216,3,16) and q.shape==(18,16) and coef.dtype==q.dtype==np.float64 and np.isfinite(coef).all() and np.isfinite(q).all()
    weights=npz(A/'pilot/weights.npz');Wo=weights['Wo'];pilotx=npz(A/'pilot/inputs.npz')['x'];pilotout=npz(A/'pilot/candidate.npz')
    hashes=[d['lookup_coefficients']['sha256'],d['separable_coefficients']['sha256']]
    assert d['pilot_reuse']==dict(input_sha256=sha(A/'pilot/inputs.npz'),candidate_sha256=sha(A/'pilot/candidate.npz'),vectors=1024,original_gate_sha256=pilot['gate_sha256'])
    beta_count=0;assert len(d['interval_controls'])==4
    for ci,(k,n) in zip(d['interval_controls'],[(0,0),(0,1),(1,1),(5,10)],strict=True):beta_count+=cp(ci,k,n)
    expected_chunks=[('train',i) for i in range(1,64)]+[(name,i) for name in ('holdout1','holdout2') for i in range(64)]
    assert [(r['panel'],r['index']) for r in d['chunks']]==expected_chunks
    events=[dict(stage='started',gate_sha256=d['gate_sha256']),dict(stage='interval_controls_completed')];reports={};input_hashes=[bh(pilotx)];stationarity=[]
    for name in ('train','holdout1','holdout2'):
        initial=np.random.PCG64(p['seeds'][name]).state # deterministic initialization; no draws
        chain=d['rng_chains'][name];assert chain['initial']==initial
        xs=[];hs=[];ls=[];prev=initial
        if name=='train':
            assert initial==pilot['input_rng_before'];prev=pilot['input_rng_after'];xs=[pilotx];hs=[pilotout['hidden']];ls=[pilotout['logits']];events.append(dict(stage='pilot_prefix_reused',vectors=1024))
        assert chain['first_new']==prev
        for row in (r for r in d['chunks'] if r['panel']==name):
            assert row['complete'] and row['rng_before']==prev;valid_state(row['rng_before'],initial);valid_state(row['rng_after'],initial);assert row['rng_before']!=row['rng_after'];prev=row['rng_after']
            index=row['index'];events.append(dict(stage='chunk_started',panel=name,index=index))
            a=load(row['inputs']);b=load(row['outputs']);assert set(a)=={'x'} and set(b)=={'hidden','logits'}
            assert Path(row['inputs']['path']).name==f'{name}_{index:03d}_inputs.npz' and Path(row['outputs']['path']).name==f'{name}_{index:03d}_outputs.npz'
            x=a['x'];h=b['hidden'];l=b['logits'];assert x.shape==(1024,10) and h.shape==(1024,16) and l.shape==(1024,10)
            assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in (x,h,l)) and np.all(h>=0)
            input_hashes.append(bh(x));xs.append(x);hs.append(h);ls.append(l);events.append(dict(stage='chunk_completed',panel=name,index=index))
        assert chain['final']==prev
        x=np.concatenate(xs);h=np.concatenate(hs);logits=np.concatenate(ls);assert x.shape==(65536,10);s=metadata(x);mask=s['case']==2;X=np.column_stack((s['m1'],s['m2'],s['last']))
        counts=np.bincount(s['slot'],minlength=216).tolist();score=d['panels'][name];assert score['state_counts']==counts and score['input_ties']==0 and score['coefficient_hashes']==hashes
        if name=='train':
            events.extend([dict(stage='lookup_fit_started'),dict(stage='lookup_fit_completed'),dict(stage='separable_fit_started'),dict(stage='separable_fit_completed')])
            meta=d['lookup_metadata'];assert meta['counts']==counts and meta['ranks']==[3]*216 and meta['rcond']==1e-12 and len(meta['singular_values'])==216 and min(counts)>=3
            for k,sv in enumerate(meta['singular_values']):
                assert len(sv)==3 and all(math.isfinite(v) and v>0 for v in sv) and sv==sorted(sv,reverse=True) and sv[-1]>1e-12*sv[0]
                at=s['slot']==k;D=X[at];target=h[at];res=D.T@(D@coef[k]-target);scale=np.abs(D).T@(np.abs(D)@np.abs(coef[k])+np.abs(target));stationarity.append(float(np.max(np.abs(res)/np.maximum(1.,scale))))
            qm=d['separable_metadata'];assert qm['count']==int(mask.sum()) and qm['rank']==18 and qm['rcond']==1e-12
            assert len(qm['singular_values'])==18 and qm['singular_values']==sorted(qm['singular_values'],reverse=True) and qm['singular_values'][-1]>1e-12*qm['singular_values'][0]
            D=np.zeros((int(mask.sum()),18));rr=np.arange(len(D));D[rr,s['p2'][mask]]=s['m2'][mask];D[rr,9+s['p1'][mask]]=s['m1'][mask]-s['m2'][mask]
            res=D.T@(D@q-h[mask]);scale=np.abs(D).T@(np.abs(D)@np.abs(q)+np.abs(h[mask]));stationarity.append(float(np.max(np.abs(res)/np.maximum(1.,scale))))
            assert max(stationarity)<1e-8 # diagnostic stationarity, no new solve/SVD or optimality certificate
        # Independent three-term/sparse-two-term prediction, no author einsum/design helper.
        B=coef[s['slot']];lh=X[:,0,None]*B[:,0,:]+X[:,1,None]*B[:,1,:]+X[:,2,None]*B[:,2,:];ll=lh@Wo.T
        sh=s['m2'][mask,None]*q[s['p2'][mask]]+(s['m1'][mask]-s['m2'][mask])[:,None]*q[9+s['p1'][mask]];sl=sh@Wo.T
        network=np.argmax(logits,axis=1);lookup=np.argmax(ll,axis=1);sep=np.argmax(sl,axis=1)
        labels=load(score['labels_archive']);expected_labels=dict(true_labels=s['true'],network=network,lookup=lookup,separable_case2_or_minus1=np.full(65536,-1,np.int64),case=s['case'],positive_bin=s['positive_bin'],state_key=s['key']);expected_labels['separable_case2_or_minus1'][mask]=sep
        assert set(labels)==set(expected_labels)
        for k,v in expected_labels.items():assert labels[k].dtype==np.int64 and np.array_equal(labels[k],v),(name,k,'label/metadata mismatch; preserve possible arithmetic-order issue')
        held=name!='train';table,beta=classify(score['network'],x,s,network,network,held);beta_count+=beta
        _,beta=classify(score['lookup'],x,s,lookup,network,held);beta_count+=beta
        subset={k:v[mask] for k,v in s.items()};_,beta=classify(score['separable_case2'],x[mask],subset,sep,network[mask],held);beta_count+=beta
        metrics=dict(lookup=hidden_metrics(score['lookup_hidden'],h,lh),separable_case2=hidden_metrics(score['separable_case2_hidden'],h[mask],sh))
        ties={key:int((np.sort(value,axis=1)[:,-1]==np.sort(value,axis=1)[:,-2]).sum()) for key,value in [('network',logits),('lookup',ll),('separable_case2',sl)]};assert ties==score['output_ties']
        wrong=network!=s['true'];blind=int((wrong&(s['positive_bin']<2)).sum());rank=1+(x>x[np.arange(len(x)),network,None]).sum(axis=1);third=int((wrong&(s['positive_bin']==2)&(s['case']<2)&(rank==3)).sum());errors=int(wrong.sum())
        assert score['disjoint_network_error_decomposition']==dict(total_errors=errors,blind_K_le_1_errors=blind,nonblind_case01_global_third_errors=third,other_nonblind_errors=errors-blind-third,denominator=65536)
        for k,row in enumerate(score['blind_conditional_errors']):
            at=s['positive_bin']==k;n=int(at.sum());e=int(wrong[at].sum());assert row['K']==k and row['n']==n and row['errors']==e and row['error_rate']==(e/n if n else None)
            if held:beta_count+=cp(row['interval'],e,n)
            else:assert row['interval']is None
        events.append(dict(stage='panel_scored',panel=name));reports[name]=dict(metrics=metrics,task_counts={k:[score[k]['task_correct'],score[k]['n']] for k in ('network','lookup','separable_case2')},disjoint_errors=score['disjoint_network_error_decomposition'])
        del x,h,logits,xs,hs,ls,B,lh,ll,sh,sl,X,s,subset
    assert len(input_hashes)==192 and len(set(input_hashes))==192
    assert d['counts']==dict(new_vectors=195584,new_normal_values=1955840,forward_attempts=191,forward_completed=191,pilot_prefix_reuses=1,fit_batches_attempted=2,fit_batches_completed=2,binomial_requests=20,beta_ppf_calls=beta_count)
    assert d['core_calls']==dict(lstsq=217,forward_batches=191,forward_sequences=195584)
    assert d['total_distinct_inputs_including_pilot']==d['total_candidate_sequences_including_pilot']==196608 and d['prior_library_sequences']==1024 and d['new_library_sequences']==0
    events.append(dict(stage='completed',results_sha256=sha(A/'fullfit/results.json')));assert events==[json.loads(v) for v in (A/'fullfit/progress.jsonl').read_text().splitlines()]
    assert d['cpu_seconds']<90 and d['maxrss_kib']<2*1024*1024
    return dict(pass_all=True,panels=reports,chunks=191,pilot_reuses=1,observations=196608,progress_records=len(events),maximum_scaled_normal_equation_residual=max(stationarity),beta_calls_reproduced=beta_count,author_cpu_seconds=d['cpu_seconds'],reviewer_refits_SVD_RNN_reference_normal_draws=0,limitations=['Saved singular values checked structurally, not independently factorized','Normal-equation residual is a numerical diagnostic, not outward optimum/rank certificate','Terminal RNG advances and neural hidden states not rerun','Prediction/scoring reconstruction is reviewer work, not a new trained fit','Individual fixed-fit CIs are numerical, not simultaneous or training-variability bounds','No historical provenance or ARC cost/rung claim'])
def main():
    out=HERE/'results.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(19,20));start=time.process_time();r=dict(complete=False,checker_sha256=sha(__file__),pins_sha256=sha(HERE/'pins.json'));out.write_text(json.dumps(r,indent=2)+'\n')
    try:r.update(check(),complete=True)
    except BaseException as e:r.update(failure=repr(e),traceback=traceback.format_exc());raise
    finally:r['review_cpu_seconds_excluding_imports']=time.process_time()-start;out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('complete','chunks','observations','review_cpu_seconds_excluding_imports')}))
if __name__=='__main__':main()
