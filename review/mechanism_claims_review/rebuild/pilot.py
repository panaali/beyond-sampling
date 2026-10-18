"""One frozen checkpoint wiring pilot; retained prefix, no surrogate fitting."""
from pathlib import Path
import hashlib, importlib.util, json, math, resource, sys, time, traceback

H=Path(__file__).resolve().parent
OUT=H/'pilot'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())

def main():
    assert not (OUT/'results.json').exists() and not (OUT/'progress.jsonl').exists()
    start=time.process_time(); OUT.mkdir(exist_ok=True)
    result=dict(complete=False,counts=dict(input_vectors=0,input_normal_values=0,candidate_attempts=0,candidate_completed=0,library_attempts=0,library_completed=0,library_hidden_hook_calls=0,surrogate_fits=0))
    def emit(stage,**fields):
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**fields),allow_nan=False)+'\n');f.flush()
    def save():
        result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=OUT/'results.tmp';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');p.replace(OUT/'results.json')
    def cap():
        if time.process_time()-start>90:raise RuntimeError('90 CPU soft cap; preserve partial, no automatic retry')
    def archive(name,**arrays):
        path=OUT/name
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        record=dict(sha256=sha(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for k,v in arrays.items()})
        result.setdefault('archives',{})[name]=record;save();return record
    try:
        gate=read(H/'pilot_gate.json');proto=read(H/'PILOT_PROTOCOL.json')
        assert gate['complete'] and gate['protocol_sha256']==sha(H/'PILOT_PROTOCOL.json')
        for p,h in {**proto['pins'],**gate['pins']}.items():assert sha(p)==h,p
        result.update(gate_sha256=sha(H/'pilot_gate.json'),protocol_sha256=sha(H/'PILOT_PROTOCOL.json'))
        emit('started',gate_sha256=result['gate_sha256']);save()
        import numpy as np
        import torch as th
        import core as C
        assert np.__version__==proto['numpy_version'] and th.__version__==proto['torch_version'] and sys.version==proto['python_version']
        th.set_num_threads(1);th.set_num_interop_threads(1)
        assert th.get_num_threads()==th.get_num_interop_threads()==1
        cap();result['import_cpu_seconds']=time.process_time()-start
        result.update(checkpoint_sha256=sha(proto['checkpoint']),runtime_versions=dict(numpy=np.__version__,torch=th.__version__,python=sys.version),architecture_sha256=sha(proto['architecture']))
        save()
        state=th.load(proto['checkpoint'],map_location='cpu',weights_only=True)
        expected={'rnn.weight_ih_l0':(16,1),'rnn.weight_hh_l0':(16,16),'linear.weight':(10,16)}
        assert set(state)==set(expected)
        assert all(tuple(state[k].shape)==shape and state[k].dtype==th.float32 and state[k].device.type=='cpu' and bool(th.isfinite(state[k]).all()) for k,shape in expected.items())
        result['stored_weights']={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()) for k,v in state.items()}
        Wi=state['rnn.weight_ih_l0'].detach().numpy()[:,0].astype(np.float64)
        Wh=state['rnn.weight_hh_l0'].detach().numpy().astype(np.float64)
        Wo=state['linear.weight'].detach().numpy().astype(np.float64)
        archive('weights.npz',Wi=Wi,Wh=Wh,Wo=Wo)
        # Original architecture, loaded without package initialization/network I/O.
        spec=importlib.util.spec_from_file_location('pinned_algzoo_architecture',proto['architecture'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng0=th.get_rng_state().clone()
        archive('torch_rng_before.npz',before=rng0.numpy())
        with th.device('meta'):model=module.DistRNN(hidden_size=16,seq_len=10,bias=False)
        rng1=th.get_rng_state().clone()
        archive('torch_rng_meta.npz',after_meta=rng1.numpy())
        model.load_state_dict(state,strict=True,assign=True)
        model=model.to(device='cpu',dtype=th.float64).eval()
        rng2=th.get_rng_state().clone()
        result['torch_rng_sha256']=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in [rng0,rng1,rng2]]
        archive('torch_rng_construction.npz',before=rng0.numpy(),after_meta=rng1.numpy(),after_assignment=rng2.numpy())
        assert th.equal(rng0,rng1) and th.equal(rng0,rng2)
        assert all(p.device.type=='cpu' and p.dtype==th.float64 for p in model.parameters())
        for k,v in model.state_dict().items():assert th.equal(v,state[k].to(th.float64)),k
        emit('checkpoint_loaded',checkpoint_sha256=sha(proto['checkpoint']));save();cap()
        rng=np.random.Generator(np.random.PCG64(proto['seeds']['train']))
        result['input_rng_before']=rng.bit_generator.state
        x=rng.standard_normal((1024,10),dtype=np.float64)
        result['counts'].update(input_vectors=1024,input_normal_values=10240)
        result['input_rng_after']=rng.bit_generator.state
        archive('inputs.npz',x=x)
        s=C.states(x) # Strict tie guard; no exclusions or resampling.
        result['state_counts']=np.bincount(s['slot'],minlength=216).tolist()
        result['input_ties']=0
        emit('inputs_saved',vectors=1024);save();cap()
        result['counts']['candidate_attempts']+=1;save();emit('candidate_started')
        hidden=C.forward_final(x,Wi,Wh);logits=hidden@Wo.T
        labels=np.argmax(logits,axis=1);ordered=np.sort(logits,axis=1);gaps=ordered[:,-1]-ordered[:,-2]
        result['counts']['candidate_completed']+=1
        archive('candidate.npz',hidden=hidden,logits=logits,labels=labels,gaps=gaps,true_labels=s['target'])
        assert np.isfinite(hidden).all() and np.isfinite(logits).all()
        emit('candidate_completed');cap()
        captured=[]
        def hook(_module,_args,output):
            result['counts']['library_hidden_hook_calls']+=1
            captured.append(output[1].detach().cpu().numpy().copy())
        handle=model.rnn.register_forward_hook(hook)
        result['counts']['library_attempts']+=1;save();emit('library_started')
        try:
            with th.no_grad():
                lx=model(th.from_numpy(x),init_state=th.zeros((1,1024,16),dtype=th.float64)).detach().cpu().numpy().copy()
        finally:handle.remove()
        result['counts']['library_completed']+=1
        assert len(captured)==1 and captured[0].shape==(1,1024,16)
        lh=captured[0][0];lp=np.argmax(lx,axis=1);ordered=np.sort(lx,axis=1);lg=ordered[:,-1]-ordered[:,-2]
        archive('library.npz',hidden=lh,logits=lx,labels=lp,gaps=lg)
        assert np.isfinite(lh).all() and np.isfinite(lx).all()
        result['torch_rng_after_forward_sha256']=hashlib.sha256(th.get_rng_state().numpy().tobytes()).hexdigest()
        archive('torch_rng.npz',before=rng0.numpy(),after_meta=rng1.numpy(),after_assignment=rng2.numpy(),after_forward=th.get_rng_state().numpy())
        assert th.equal(rng0,th.get_rng_state())
        metrics={}
        for name,left,right in [('hidden',hidden,lh),('logits',logits,lx)]:
            error=float(np.max(np.abs(left-right)));scale=max(1.,float(np.max(np.abs(right))))
            metrics[name]=dict(maximum_absolute_error=error,reference_scale=scale,tolerance=1e-10*scale,passes=error<=1e-10*scale)
        mismatch=np.flatnonzero(labels!=lp).tolist()
        result.update(comparison=metrics,label_mismatch_indices=mismatch,candidate_output_ties=int(np.count_nonzero(gaps==0)),library_output_ties=int(np.count_nonzero(lg==0)),candidate_core_calls=C.CALLS.copy())
        emit('library_completed',label_mismatches=len(mismatch));save();cap()
        assert all(m['passes'] for m in metrics.values()) and not mismatch
        assert C.CALLS==dict(lstsq=0,forward_batches=1,forward_sequences=1024)
        assert result['counts']==dict(input_vectors=1024,input_normal_values=10240,candidate_attempts=1,candidate_completed=1,library_attempts=1,library_completed=1,library_hidden_hook_calls=1,surrogate_fits=0)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024*1024
        result.update(complete=True,scope=proto['scope'],future_reuse='Load saved x/hidden/logits once as training prefix; resume saved PCG64 state for remaining64512 inputs. No new forward of prefix.')
        save();emit('completed',results_sha256=sha(OUT/'results.json'))
        print(json.dumps(dict(complete=True,comparison=metrics,counts=result['counts'],cpu=result['cpu_seconds'])))
    except BaseException:
        result['failure']=traceback.format_exc();save();emit('failure',counts=result['counts']);raise

if __name__=='__main__':main()
