"""Frozen random-He-width diagnostic. Candidate never sees random model inputs."""
import argparse,hashlib,json,math,resource,time,traceback
import numpy as np
from mixture import HERE,components,combine,continuation,prefix2,rnn_step_inject
R=json.loads((HERE/'random_protocol.json').read_text())

def candidate(wi,wh,method):
    h=len(wi);q=wi@wi.T;mu=np.zeros(h);cov=np.zeros((h,h));meta={}
    if method=='sector_mixture':
        trace,info=continuation(wi[:,0],wh);mu,cov=trace[-1]
        meta=dict(components=len(info['weights']),closure_calls=info['closure_calls'],weights=info['weights'],final_component_means=info['states'][-1]['mean'],final_component_covariances=info['states'][-1]['covariance'])
    else:
        for t in range(1,5):
            if method=='collapsed_h2' and t==2:mu,cov,_=prefix2(wi[:,0],wh)
            else:mu,cov=rnn_step_inject(mu,cov,wh,q,'arc2' if method=='arc2' else 'gauss')
        meta=dict(components=1,closure_calls=3 if method=='collapsed_h2' else 4)
    return mu,cov,meta

def cell(h,d,seed):
    start=time.process_time();rng=np.random.default_rng(seed);wi=rng.standard_normal((h,d))*math.sqrt(2/d);wh=rng.standard_normal((h,h))*math.sqrt(2/h);wo=rng.standard_normal((h,h))*math.sqrt(1/h)
    outputs={};states={};methods=R['scalar_methods'] if d==1 else R['vector_methods']
    for meth in methods:
        t=time.process_time();mu,cov,meta=candidate(wi,wh,meth);cpu=time.process_time()-t
        outputs[meth]=dict(output_mean=(wo@mu).tolist(),hidden_mean=mu.tolist(),hidden_covariance=cov.tolist(),cpu_seconds=cpu,components=meta['components'],closure_calls=meta['closure_calls'])
        if meth=='sector_mixture':states=dict(component_weights=np.array(meta['weights']),component_means=np.array(meta['final_component_means']),component_covariances=np.array(meta['final_component_covariances']))
    reference_seed=202610170690+1000*h+10*d+seed;rng=np.random.default_rng(reference_seed);n=R['reference_block_size'];bm=[];bs=[];t=time.process_time()
    for b in range(R['reference_blocks']):
        hidden=np.zeros((n,h))
        for step in range(4):
            hidden=np.maximum(hidden@wh.T+rng.standard_normal((n,d))@wi.T,0)
        y=hidden@wo.T;bm.append(y.sum(0)/n);bs.append((y*y).sum(0)/n)
    ref_cpu=time.process_time()-t;bm=np.array(bm);bs=np.array(bs);mean=bm.mean(0);var=bs.mean(0)-mean*mean;bvar=bs-bm*bm
    assert np.all(var>0),var
    for meth,out in outputs.items():
        prediction=np.array(out['output_mean']);err=(prediction-mean)**2/var;block=np.mean((prediction-bm)**2/bvar,axis=1)
        out.update(per_output_vMSE=err.tolist(),mean_vMSE=float(err.mean()),reference_noise_floor=1/R['reference_samples'],floor_subtracted_vMSE=float(err.mean()-1/R['reference_samples']),block_vMSE=block.tolist(),block_vMSE_sd=float(block.std(ddof=1)),mean_error_block_standard_error=(bm.std(0,ddof=1)/np.sqrt(len(bm))).tolist())
    stem=f'random_H{h}_d{d}_s{seed}';np.savez_compressed(HERE/(stem+'.npz'),wi=wi,wh=wh,wo=wo,reference_block_mean=bm,reference_block_raw_second=bs,reference_mean=mean,reference_variance=var,**states)
    # Major named scalar source kernels only; RNG/control/moment reductions listed separately.
    N=R['reference_samples'];cost=dict(reference_gaussian_scalars=N*4*d,reference_recurrent_muladd=N*4*h*(2*h-1),reference_input_muladd=N*4*h*(2*d-1),reference_vector_add=N*4*h,reference_relu_comparison=N*4*h,reference_readout_muladd=N*h*(2*h-1),reference_squares=N*h,qualification='Named main source kernels only, excludes RNG internals/aggregate divisions/reductions/runtime. Source counts and CPU are not a full competitive-cost claim.')
    return dict(complete=True,H=h,d=d,model_seed=seed,reference_seed=reference_seed,weight_sha256=hashlib.sha256(wi.tobytes()+wh.tobytes()+wo.tobytes()).hexdigest(),archive_sha256=hashlib.sha256((HERE/(stem+'.npz')).read_bytes()).hexdigest(),methods=outputs,reference_cpu_seconds=ref_cpu,cpu_seconds=time.process_time()-start,compute_accounting=cost)

def main():
    p=argparse.ArgumentParser();p.add_argument('H',type=int,choices=R['widths']);h=p.parse_args().H;resource.setrlimit(resource.RLIMIT_CPU,(110,110));start=time.process_time()
    for d in (1,h):
        for seed in R['model_seeds']:
            path=HERE/f'random_H{h}_d{d}_s{seed}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['complete'];continue
            try:out=cell(h,d,seed)
            except Exception as exc:
                out=dict(complete=False,H=h,d=d,model_seed=seed,error=repr(exc),traceback=traceback.format_exc(),cpu_seconds=time.process_time()-start);path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');raise
            out.update(protocol_sha256=hashlib.sha256((HERE/'random_protocol.json').read_bytes()).hexdigest(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(H=h,d=d,seed=seed,cpu_seconds=out['cpu_seconds'],vMSE={m:o['mean_vMSE'] for m,o in out['methods'].items()})),flush=True)
    (HERE/f'random_H{h}_completion.json').write_text(json.dumps(dict(complete=True,H=h,cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
if __name__=='__main__':main()
