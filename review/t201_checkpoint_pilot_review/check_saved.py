"""Saved pilot contraction; no Torch/checkpoint load/forward/normal draws."""
from pathlib import Path
import hashlib,json,math,time,resource,traceback
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_78/review/mechanism_claims_review/rebuild')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bytes_sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def check():
    pins=read(HERE/'saved_pins.json')
    for path,h in pins['pins'].items():assert sha(path)==h,path
    p=read(A/'PILOT_PROTOCOL.json');g=read(A/'pilot_gate.json');d=read(A/'pilot/results.json')
    assert np.__version__==p['numpy_version'] and d['complete'] and g['complete']
    assert d['protocol_sha256']==g['protocol_sha256']==sha(A/'PILOT_PROTOCOL.json') and d['gate_sha256']==sha(A/'pilot_gate.json')
    for path,h in {**p['pins'],**g['pins']}.items():assert sha(path)==h,path
    assert d['checkpoint_sha256']==sha(p['checkpoint']) and d['architecture_sha256']==sha(p['architecture'])
    assert d['runtime_versions']==dict(numpy=p['numpy_version'],torch=p['torch_version'],python=p['python_version'])
    assert d['scope']==p['scope'] and len(set(p['seeds'].values()))==3
    expected_files={'weights.npz','inputs.npz','candidate.npz','library.npz','torch_rng_before.npz','torch_rng_meta.npz','torch_rng_construction.npz','torch_rng.npz'}
    assert set(d['archives'])==expected_files
    arrays={}
    for name,meta in d['archives'].items():
        path=A/'pilot'/name;assert sha(path)==meta['sha256']
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(meta['arrays']);arrays[name]={k:z[k].copy() for k in z.files}
        for key,x in arrays[name].items():assert dict(shape=list(x.shape),dtype=str(x.dtype),bytes_sha256=bytes_sha(x))==meta['arrays'][key]
    weights=arrays['weights.npz'];assert set(weights)=={'Wi','Wh','Wo'}
    mapping={'Wi':('rnn.weight_ih_l0',(16,),(16,1)),'Wh':('rnn.weight_hh_l0',(16,16),(16,16)),'Wo':('linear.weight',(10,16),(10,16))}
    for name,(key,shape,stored_shape) in mapping.items():
        x=weights[name];assert x.dtype==np.float64 and x.shape==shape and np.isfinite(x).all()
        old=x.astype(np.float32).reshape(stored_shape)
        assert np.array_equal(old.reshape(shape).astype(np.float64),x)
        assert d['stored_weights'][key]==dict(shape=list(stored_shape),dtype='torch.float32',sha256=bytes_sha(old))
    assert set(d['stored_weights'])=={x[0] for x in mapping.values()}
    x=arrays['inputs.npz']['x'];assert x.shape==(1024,10) and x.dtype==np.float64 and np.isfinite(x).all()
    start=np.random.PCG64(p['seeds']['train']).state # Deterministic initialization only: no draws.
    assert d['input_rng_before']==start
    after=d['input_rng_after'];assert set(after)==set(start) and after['bit_generator']=='PCG64'
    assert set(after['state'])=={'state','inc'} and after['state']['inc']==start['state']['inc']
    assert type(after['state']['state'])is int and 0<=after['state']['state']<2**128 and after['state']['state']!=start['state']['state']
    assert type(after['has_uint32'])is int and after['has_uint32']in(0,1) and type(after['uinteger'])is int
    states=Counter();truth=[];legal={(c,a,b):j for j,(c,a,b) in enumerate((c,a,b) for c in range(3) for a in range(9) for b in range(9) if a!=b)}
    for row in x.tolist():
        assert len(set(row))==10
        order=sorted(range(10),key=lambda j:(-row[j],j));truth.append(order[1])
        head=sorted(range(9),key=lambda j:(-row[j],j));a,b=head[:2];c=0 if row[9]>row[a] else 1 if row[9]>row[b] else 2
        states[legal[(c,a,b)]]+=1
    assert d['state_counts']==[states[i] for i in range(216)] and sum(d['state_counts'])==1024 and d['input_ties']==0
    cand=arrays['candidate.npz'];lib=arrays['library.npz'];assert set(cand)=={'hidden','logits','labels','gaps','true_labels'} and set(lib)=={'hidden','logits','labels','gaps'}
    assert cand['true_labels'].tolist()==truth
    for a in (cand,lib):
        for key,shape in [('hidden',(1024,16)),('logits',(1024,10)),('gaps',(1024,))]:assert a[key].shape==shape and a[key].dtype==np.float64 and np.isfinite(a[key]).all()
        assert a['labels'].shape==(1024,) and a['labels'].dtype==np.int64
        labels=[];gaps=[]
        for row in a['logits'].tolist():
            labels.append(max(range(10),key=lambda j:row[j]));order=sorted(row);gaps.append(order[-1]-order[-2])
        assert labels==a['labels'].tolist() and gaps==a['gaps'].tolist()
    metrics={}
    for key in ('hidden','logits'):
        left=cand[key].ravel().tolist();right=lib[key].ravel().tolist()
        error=max(abs(a-b) for a,b in zip(left,right,strict=True));scale=max(1.0,max(map(abs,right)));tol=1e-10*scale
        metrics[key]=dict(maximum_absolute_error=error,reference_scale=scale,tolerance=tol,passes=error<=tol)
    assert d['comparison']==metrics and all(v['passes'] for v in metrics.values())
    mismatches=[i for i,(a,b) in enumerate(zip(cand['labels'].tolist(),lib['labels'].tolist(),strict=True)) if a!=b]
    assert d['label_mismatch_indices']==mismatches==[]
    assert d['candidate_output_ties']==sum(v==0 for v in cand['gaps'].tolist()) and d['library_output_ties']==sum(v==0 for v in lib['gaps'].tolist())
    rng=arrays['torch_rng.npz'];assert set(rng)=={'before','after_meta','after_assignment','after_forward'}
    for value in rng.values():assert value.dtype==np.uint8 and value.ndim==1 and np.array_equal(value,rng['before'])
    assert arrays['torch_rng_before.npz']['before'].tobytes()==rng['before'].tobytes()
    assert arrays['torch_rng_meta.npz']['after_meta'].tobytes()==rng['after_meta'].tobytes()
    construction=arrays['torch_rng_construction.npz'];assert set(construction)=={'before','after_meta','after_assignment'}
    for k,v in construction.items():assert v.tobytes()==rng[k].tobytes()
    assert d['torch_rng_sha256']==[bytes_sha(rng[k]) for k in ('before','after_meta','after_assignment')]
    assert d['torch_rng_after_forward_sha256']==bytes_sha(rng['after_forward'])
    counts=dict(input_vectors=1024,input_normal_values=10240,candidate_attempts=1,candidate_completed=1,library_attempts=1,library_completed=1,library_hidden_hook_calls=1,surrogate_fits=0)
    assert d['counts']==counts and d['candidate_core_calls']==dict(lstsq=0,forward_batches=1,forward_sequences=1024)
    events=[json.loads(line) for line in (A/'pilot/progress.jsonl').read_text().splitlines()]
    expected=[dict(stage='started',gate_sha256=d['gate_sha256']),dict(stage='checkpoint_loaded',checkpoint_sha256=d['checkpoint_sha256']),dict(stage='inputs_saved',vectors=1024),dict(stage='candidate_started'),dict(stage='candidate_completed'),dict(stage='library_started'),dict(stage='library_completed',label_mismatches=0),dict(stage='completed',results_sha256=sha(A/'pilot/results.json'))]
    assert events==expected
    assert 0<=d['import_cpu_seconds']<=d['cpu_seconds']<90 and d['maxrss_kib']<2*1024*1024
    return dict(pass_all=True,arrays=len(d['archives']),observations=1024,comparison=metrics,labels_match=True,state_count_sum=sum(states.values()),author_counts=counts,author_cpu_seconds=d['cpu_seconds'],author_maxrss_kib=d['maxrss_kib'],reviewer_checkpoint_load_forward_fits_random_draws=0,scope='Saved array/pin/RNG/parity audit only. Initial PCG64 state reconstructed without draws; terminal stream advance not replayed. Stored float32 conversion hash checked against source-gated author checkpoint records, no independent checkpoint deserialization. No heldout/historical/performance claim.')
def main():
    out=HERE/'saved_results.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(19,20));start=time.process_time();r=dict(complete=False,checker_sha256=sha(__file__),pins_sha256=sha(HERE/'saved_pins.json'));out.write_text(json.dumps(r,indent=2)+'\n')
    try:r.update(check(),complete=True)
    except BaseException as e:r.update(failure=repr(e),traceback=traceback.format_exc());raise
    finally:r['review_cpu_seconds']=time.process_time()-start;out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
if __name__=='__main__':main()
