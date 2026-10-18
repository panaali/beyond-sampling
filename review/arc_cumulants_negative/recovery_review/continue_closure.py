"""Frozen one-model exact-h2 reset vs original Gaussian propagation; MC is reference only."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from check_prefix_covariance import load_model
from prefix_covariance import prefix2
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'rescue_protocol.json').read_text())
assert hashlib.sha256((HERE/'cumprop_snapshot.py').read_bytes()).hexdigest()==P['source_sha256']
from cumprop_snapshot import rnn_step_inject

def metrics(mu,cov,rm,rc):
    off=~np.eye(len(mu),dtype=bool)
    rms=lambda a:float(np.sqrt(np.mean(np.asarray(a)**2)))
    return dict(mean=rms(mu-rm)/rms(rm),variance=rms(np.diag(cov-rc))/rms(np.diag(rc)),off_diagonal=rms((cov-rc)[off])/rms(rc[off]),covariance=rms(cov-rc)/rms(rc))

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(110,110));start=time.process_time();assert json.loads((HERE/'covariance_controls.json').read_text())['complete'];assert json.loads((HERE/'covariance_trained.json').read_text())['complete']
    wi,wh=load_model();h=len(wi);Q=np.outer(wi,wi);traces={};times={}
    for name in P['methods']:
        t0=time.process_time();mu=np.zeros(h);cov=np.zeros((h,h));rows=[]
        for t in range(1,5):
            if name=='exact_h2_then_inherited_gauss' and t==2:mu,cov,metadata=prefix2(wi,wh)
            else:mu,cov=rnn_step_inject(mu,cov,wh,Q,'gauss')
            rows.append((mu.copy(),cov.copy()))
        traces[name]=rows;times[name]=time.process_time()-t0
    rng=np.random.default_rng(P['mc_seed']);n=P['mc_block_size'];block_means=[];block_raw=[];t0=time.process_time()
    for block in range(P['mc_blocks']):
        x=rng.standard_normal((n,4));hidden=np.zeros((n,h));mm=[];rr=[]
        for t in range(4):
            hidden=np.maximum(hidden@wh.T+x[:,t:t+1]*wi[None,:],0)
            mm.append(hidden.sum(0)/n);rr.append((hidden.T@hidden)/n)
        block_means.append(mm);block_raw.append(rr)
    reference_cpu=time.process_time()-t0
    bm=np.array(block_means);br=np.array(block_raw);bc=br-bm[:,:,:,None]*bm[:,:,None,:]
    rm=bm.mean(0);rr=br.mean(0);rc=rr-rm[:,:,None]*rm[:,None,:]
    np.savez_compressed(HERE/'closure_reference.npz',block_means=bm,block_raw_second=br,mean=rm,covariance=rc,mean_block_se=bm.std(0,ddof=1)/np.sqrt(len(bm)),covariance_block_se=bc.std(0,ddof=1)/np.sqrt(len(bm)))
    rows=[]
    for t in range(4):
        row=dict(step=t+1,methods={})
        for name,trace in traces.items():
            mu,cov=trace[t];row['methods'][name]=dict(mean=mu.tolist(),covariance=cov.tolist(),relative_rms_errors=metrics(mu,cov,rm[t],rc[t]),block_errors=[metrics(mu,cov,bm[b,t],bc[b,t]) for b in range(len(bm))],min_covariance_eigenvalue=float(np.linalg.eigvalsh(cov)[0]))
        improvements={}
        for key in ('mean','variance','off_diagonal','covariance'):
            a=row['methods'][P['methods'][0]];b=row['methods'][P['methods'][1]];d=np.array([aa[key]-bb[key] for aa,bb in zip(a['block_errors'],b['block_errors'])])
            improvements[key]=dict(pooled_error_reduction=a['relative_rms_errors'][key]-b['relative_rms_errors'][key],block_mean_error_reduction=float(d.mean()),block_standard_error=float(d.std(ddof=1)/np.sqrt(len(d))))
        row['paired_improvements']=improvements;rows.append(row)
    total=P['mc_samples'];steps=4;blocks=P['mc_blocks']
    cost=dict(reference_random_normal_scalars=total*steps,reference_recurrent_scalar_multiply_add=total*steps*h*(2*h-1),reference_input_multiply_add=total*steps*2*h,reference_relu_comparisons=total*steps*h,reference_block_sum_additions=blocks*steps*h*(n-1),reference_raw_second_multiply_add=blocks*steps*h*h*(2*n-1),excluded_from_named_subtotal='RNG internals, division/aggregation/statistical summaries, BLAS dispatch and memory; all CPU included in process totals.',candidate_scope='Source operation structure and timings only; no full-cost claim. Inherited Gaussian rule construction and pair quadrature retained unchanged.')
    out=dict(complete=True,protocol_sha256=hashlib.sha256((HERE/'rescue_protocol.json').read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((HERE/'closure_reference.npz').read_bytes()).hexdigest(),weight_sha256=P['weight_sha256'],rows=rows,method_cpu_seconds=times,reference_cpu_seconds=reference_cpu,cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,compute_accounting=cost,qualification='Finite one-model hidden-moment diagnostic. Eight block SEs are descriptive, not rigorous confidence bounds; reference covariance uses population denominator, no reference clipping. No final accuracy or competitive ARC estimate.')
    (HERE/'closure_result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,cpu_seconds=out['cpu_seconds'],reference_cpu_seconds=reference_cpu,errors=[{n:r['methods'][n]['relative_rms_errors'] for n in P['methods']} for r in rows])))
if __name__=='__main__':main()
