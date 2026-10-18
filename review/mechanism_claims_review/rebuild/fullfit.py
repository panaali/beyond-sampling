"""Frozen reconstruction train/two-holdout experiment; no adaptive selection."""
from pathlib import Path
import hashlib,json,math,resource,sys,time,traceback
H=Path(__file__).resolve().parent
OUT=H/'fullfit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    assert not (OUT/'results.json').exists() and not (OUT/'progress.jsonl').exists()
    OUT.mkdir(exist_ok=True);start=time.process_time()
    result=dict(complete=False,chunks=[],panels={},counts=dict(new_vectors=0,new_normal_values=0,forward_attempts=0,forward_completed=0,pilot_prefix_reuses=0,fit_batches_attempted=0,fit_batches_completed=0,binomial_requests=0,beta_ppf_calls=0))
    C=None
    def save():
        result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if C is not None:result['core_calls']=C.CALLS.copy()
        p=OUT/'results.tmp';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');p.replace(OUT/'results.json')
    def emit(stage,**kw):
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**kw),allow_nan=False)+'\n');f.flush()
    def cap():
        if time.process_time()-start>90:raise RuntimeError('90 CPU total soft guard; stop, no retry')
    def archive(name,**arrays):
        path=OUT/name
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        meta=dict(path=str(path),sha256=sha(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()})
        return meta
    try:
        gate=read(H/'fullfit_gate.json');proto=read(H/'FULLFIT_PROTOCOL.json')
        assert gate['complete'] and gate['protocol_sha256']==sha(H/'FULLFIT_PROTOCOL.json')
        for p,h in {**proto['pins'],**gate['pins']}.items():assert sha(p)==h,p
        result.update(gate_sha256=sha(H/'fullfit_gate.json'),protocol_sha256=sha(H/'FULLFIT_PROTOCOL.json'))
        emit('started',gate_sha256=result['gate_sha256']);save()
        import numpy as np
        import scipy
        from scipy.stats import beta
        import core as C
        assert np.__version__==proto['numpy_version'] and scipy.__version__==proto['scipy_version'] and sys.version==proto['python_version']
        cap();result['runtime_versions']=dict(numpy=np.__version__,scipy=scipy.__version__,python=sys.version)
        def interval(k,n):
            result['counts']['binomial_requests']+=1
            assert type(k)is int and type(n)is int and 0<=k<=n
            if n==0:return None
            def q(p,a,b):
                result['counts']['beta_ppf_calls']+=1
                return float(beta.ppf(p,a,b))
            lo=0. if k==0 else q(.025,k,n-k+1)
            hi=1. if k==n else q(.975,k+1,n-k)
            assert 0<=lo<=k/n<=hi<=1
            return dict(lower=lo,upper=hi,level=.95,method='ordinary numerical equal-tail Clopper-Pearson; individual fixed-fit interval')
        ci0=interval(0,0);ci1=interval(0,1);ci2=interval(1,1);ci3=interval(5,10)
        assert ci0 is None and ci1['lower']==0 and abs(ci1['upper']-.975)<1e-15
        assert ci2['upper']==1 and abs(ci2['lower']-.025)<1e-15
        assert abs(ci3['lower']+ci3['upper']-1)<1e-14
        result['interval_controls']=[ci0,ci1,ci2,ci3];emit('interval_controls_completed');save()
        pilot=read(H/'pilot/results.json');assert pilot['complete']
        assert pilot['protocol_sha256']==sha(H/'PILOT_PROTOCOL.json')
        assert proto['seeds']==read(H/'PILOT_PROTOCOL.json')['seeds']
        with np.load(H/'pilot/weights.npz',allow_pickle=False) as z:Wi=z['Wi'].copy();Wh=z['Wh'].copy();Wo=z['Wo'].copy()
        assert Wi.shape==(16,) and Wh.shape==(16,16) and Wo.shape==(10,16)
        assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in [Wi,Wh,Wo])
        with np.load(H/'pilot/inputs.npz',allow_pickle=False) as z:px=z['x'].copy()
        with np.load(H/'pilot/candidate.npz',allow_pickle=False) as z:ph=z['hidden'].copy();pl=z['logits'].copy()
        assert px.shape==(1024,10) and ph.shape==(1024,16) and pl.shape==(1024,10)
        models={};coefficients=None
        def make_panel(name):
            rng=np.random.Generator(np.random.PCG64(proto['seeds'][name]));initial=rng.bit_generator.state
            xs=[];hs=[];ls=[];first=0
            if name=='train':
                assert initial==pilot['input_rng_before']
                rng.bit_generator.state=pilot['input_rng_after']
                xs.append(px);hs.append(ph);ls.append(pl);first=1
                result['counts']['pilot_prefix_reuses']+=1
                result['pilot_reuse']=dict(input_sha256=sha(H/'pilot/inputs.npz'),candidate_sha256=sha(H/'pilot/candidate.npz'),vectors=1024,original_gate_sha256=pilot['gate_sha256'])
                emit('pilot_prefix_reused',vectors=1024);save()
            result.setdefault('rng_chains',{})[name]=dict(initial=initial,first_new=rng.bit_generator.state)
            for chunk in range(first,64):
                cap();record=dict(panel=name,index=chunk,complete=False,rng_before=rng.bit_generator.state)
                result['chunks'].append(record);emit('chunk_started',panel=name,index=chunk);save()
                x=rng.standard_normal((1024,10),dtype=np.float64)
                record['rng_after']=rng.bit_generator.state
                result['counts']['new_vectors']+=1024;result['counts']['new_normal_values']+=10240
                # Persist inputs before any forward, preserving a partial attempt.
                record['inputs']=archive(f'{name}_{chunk:03d}_inputs.npz',x=x);save()
                C.states(x) # Stop on ties; no exclusions or replacement samples.
                result['counts']['forward_attempts']+=1;save()
                hidden=C.forward_final(x,Wi,Wh);logits=hidden@Wo.T
                result['counts']['forward_completed']+=1
                record['outputs']=archive(f'{name}_{chunk:03d}_outputs.npz',hidden=hidden,logits=logits);save()
                assert np.isfinite(hidden).all() and np.isfinite(logits).all() and np.all(hidden>=0)
                record['complete']=True;xs.append(x);hs.append(hidden);ls.append(logits)
                emit('chunk_completed',panel=name,index=chunk);save()
            result['rng_chains'][name]['final']=rng.bit_generator.state
            return np.concatenate(xs),np.concatenate(hs),np.concatenate(ls)
        def classification(s,labels,network_labels,heldout):
            n=len(labels);correct=int(np.count_nonzero(labels==s['target']));agree=int(np.count_nonzero(labels==network_labels))
            tab=C.cross_tab(s,labels);assert tab['correct']==correct and tab['n']==n
            return dict(n=n,task_correct=correct,network_agree=agree,task_accuracy=correct/n,network_agreement=agree/n,
                        task_interval=interval(correct,n) if heldout else None,agreement_interval=interval(agree,n) if heldout else None,
                        interval_scope='fixed model and fixed fitted coefficients; individual, not simultaneous' if heldout else 'training data: no fixed-fit binomial interval',cross_tab=tab)
        for name in ['train','holdout1','holdout2']:
            x,h,logits=make_panel(name);assert x.shape==(65536,10) and h.shape==(65536,16) and logits.shape==(65536,10)
            s=C.states(x);network=np.argmax(logits,axis=1);mask=s['case']==2
            assert mask.any()
            if name=='train':
                emit('lookup_fit_started');result['counts']['fit_batches_attempted']+=1;save();cap()
                B,bmeta=C.fit_lookup(s,h);result['counts']['fit_batches_completed']+=1
                result['lookup_coefficients']=archive('lookup_coefficients.npz',coefficients=B);result['lookup_metadata']=bmeta;save();emit('lookup_fit_completed');assert np.isfinite(B).all()
                emit('separable_fit_started');result['counts']['fit_batches_attempted']+=1;save();cap()
                Q,qmeta=C.fit_separable(s,h);result['counts']['fit_batches_completed']+=1
                result['separable_coefficients']=archive('separable_coefficients.npz',coefficients=Q);result['separable_metadata']=qmeta;save();emit('separable_fit_completed')
                assert np.isfinite(B).all() and np.isfinite(Q).all()
                coefficients=(sha(OUT/'lookup_coefficients.npz'),sha(OUT/'separable_coefficients.npz'))
            assert coefficients==(sha(OUT/'lookup_coefficients.npz'),sha(OUT/'separable_coefficients.npz'))
            cap();lh=C.predict_lookup(s,B);ll=lh@Wo.T;look=np.argmax(ll,axis=1)
            D=C.separable_design(s);sh=D[mask]@Q;sl=sh@Wo.T;sep=np.argmax(sl,axis=1)
            assert all(np.isfinite(a).all() for a in [lh,ll,sh,sl])
            subset={k:v[mask] for k,v in s.items()};heldout=name!='train'
            score=dict(network=classification(s,network,network,heldout),lookup=classification(s,look,network,heldout),separable_case2=classification(subset,sep,network[mask],heldout),
                       lookup_hidden=C.hidden_metrics(h,lh),separable_case2_hidden=C.hidden_metrics(h[mask],sh),state_counts=np.bincount(s['slot'],minlength=216).tolist(),
                       coefficient_hashes=list(coefficients),input_ties=0,output_ties=dict(network=int(np.count_nonzero(np.sort(logits,axis=1)[:,-1]==np.sort(logits,axis=1)[:,-2])),lookup=int(np.count_nonzero(np.sort(ll,axis=1)[:,-1]==np.sort(ll,axis=1)[:,-2])),separable_case2=int(np.count_nonzero(np.sort(sl,axis=1)[:,-1]==np.sort(sl,axis=1)[:,-2]))))
            table=np.array(score['network']['cross_tab']['table'],dtype=np.int64)
            errors=65536-score['network']['task_correct'];blind=int(table[:2].sum()-table[:2,:,1].sum());third=int(table[2,:2,2].sum())
            other=errors-blind-third;assert min(blind,third,other)>=0
            score['disjoint_network_error_decomposition']=dict(total_errors=errors,blind_K_le_1_errors=blind,nonblind_case01_global_third_errors=third,other_nonblind_errors=other,denominator=65536)
            score['blind_conditional_errors']=[]
            for k in [0,1]:
                n=int(table[k].sum());e=n-int(table[k,:,1].sum())
                score['blind_conditional_errors'].append(dict(K=k,n=n,errors=e,error_rate=e/n if n else None,interval=interval(e,n) if heldout else None))
            fullsep=np.full(65536,-1,dtype=np.int64);fullsep[mask]=sep
            score['labels_archive']=archive(f'{name}_labels.npz',true_labels=s['target'],network=network,lookup=look,separable_case2_or_minus1=fullsep,case=s['case'],positive_bin=s['positive_bin'],state_key=s['key'])
            result['panels'][name]=score;emit('panel_scored',panel=name);save();cap()
            del x,h,logits,s,lh,ll,look,D,sh,sl,sep,subset,network,mask,table
        assert len(result['chunks'])==191 and all(r['complete'] for r in result['chunks'])
        expected=[('train',i) for i in range(1,64)]+[(name,i) for name in ['holdout1','holdout2'] for i in range(64)]
        assert [(r['panel'],r['index']) for r in result['chunks']]==expected
        counts=result['counts'];assert counts['new_vectors']==195584 and counts['new_normal_values']==1955840
        assert counts['forward_attempts']==counts['forward_completed']==191 and counts['pilot_prefix_reuses']==1
        assert counts['fit_batches_attempted']==counts['fit_batches_completed']==2
        assert counts['binomial_requests']==20
        assert C.CALLS==dict(lstsq=217,forward_batches=191,forward_sequences=195584)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024*1024
        result.update(complete=True,scope=proto['scope'],total_distinct_inputs_including_pilot=196608,total_candidate_sequences_including_pilot=196608,prior_library_sequences=1024,new_library_sequences=0)
        save();emit('completed',results_sha256=sha(OUT/'results.json'))
        print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],counts=counts,panels={n:{k:d[k]['task_accuracy'] for k in ['network','lookup','separable_case2']} for n,d in result['panels'].items()})))
    except BaseException:
        result['failure']=traceback.format_exc();save();emit('failure',counts=result['counts']);raise

if __name__=='__main__':main()
