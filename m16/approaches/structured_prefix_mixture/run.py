import argparse,json,resource,time
import numpy as np
from mixture import HERE,P,SRC,components,combine,continuation,prefix2
from check_prefix_covariance import fixtures,load_model
from continue_closure import metrics

def controls():
    rows=[]
    for name,wi,wh in fixtures():
        w,m,c,metadata=components(wi,wh);mu,cov=combine(w,m,c);rm,rc,_=prefix2(wi,wh)
        valid=bool(abs(w.sum()-1)<=P['component_mass_tolerance'] and np.all(w>0) and np.allclose(mu,rm,atol=P['aggregate_abs_tolerance'],rtol=P['aggregate_rel_tolerance']) and np.allclose(cov,rc,atol=P['aggregate_abs_tolerance'],rtol=P['aggregate_rel_tolerance']))
        rows.append(dict(name=name,passed=valid,weights=w.tolist(),mean=m.tolist(),covariance=c.tolist(),metadata=metadata,aggregate_mean_maxerror=float(np.max(abs(mu-rm))),aggregate_covariance_maxerror=float(np.max(abs(cov-rc)))))
    return dict(complete=all(r['passed'] for r in rows),rows=rows)

def trained():
    assert json.loads((HERE/'controls.json').read_text())['complete'];wi,wh=load_model();start=time.process_time();trace,metadata=continuation(wi,wh);candidate_cpu=time.process_time()-start
    ref=np.load(SRC/'closure_reference.npz');bm=ref['block_means'];br=ref['block_raw_second'];bc=br-bm[:,:,:,None]*bm[:,:,None,:];rm=ref['mean'];rc=ref['covariance'];old=json.loads((SRC/'closure_result.json').read_text());rows=[]
    for t,(mu,cov) in zip((2,3,4),trace):
        errors=metrics(mu,cov,rm[t-1],rc[t-1]);blocks=[metrics(mu,cov,m[t-1],c[t-1]) for m,c in zip(bm,bc)];baseline=old['rows'][t-1]['methods']['exact_h2_then_inherited_gauss'];diff={}
        for key in errors:
            d=np.array([a[key]-b[key] for a,b in zip(baseline['block_errors'],blocks)])
            diff[key]=dict(pooled_reduction=baseline['relative_rms_errors'][key]-errors[key],block_mean_reduction=float(d.mean()),block_se=float(d.std(ddof=1)/np.sqrt(len(d))))
        rows.append(dict(step=t,mean=mu.tolist(),covariance=cov.tolist(),relative_rms_errors=errors,block_errors=blocks,paired_reduction_from_collapsed=diff,min_covariance_eigenvalue=float(np.linalg.eigvalsh(cov)[0])))
    return dict(complete=True,rows=rows,metadata=metadata,candidate_cpu_seconds=candidate_cpu,no_new_random_inputs=True,qualification='One-model hidden-moment diagnostic, ordinary floating geometry and inherited Gaussian closure; no accuracy or complete-cost assertion.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['controls','trained']);args=ap.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();out=globals()[args.mode]();out.update(cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/(args.mode+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=out['complete'],cpu_seconds=out['cpu_seconds'],errors=[r.get('relative_rms_errors',{}) for r in out['rows']])));assert out['complete']
if __name__=='__main__':main()
