import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,time,numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/structured_prefix_mixture');R=Path('/workspace/work/recover-small_h2_56/review/arc_cumulants_negative/recovery_review');start=time.process_time();d=json.loads((P/'trained.json').read_text());meta=d['metadata'];w=np.array(meta['weights']);assert len(w)==34 and np.all(w>0) and abs(sum(w)-1)<1e-15 and meta['closure_calls']==68
ref=np.load(R/'closure_reference.npz');bm=ref['block_means'];br=ref['block_raw_second'];bc=br-bm[:,:,:,None]*bm[:,:,None,:];old=json.loads((R/'closure_result.json').read_text())
def metrics(mu,cov,rm,rc):
 off=~np.eye(16,dtype=bool);rms=lambda a:float(np.sqrt(np.mean(a*a)))
 return dict(mean=rms(mu-rm)/rms(rm),variance=rms(np.diag(cov-rc))/rms(np.diag(rc)),off_diagonal=rms((cov-rc)[off])/rms(rc[off]),covariance=rms(cov-rc)/rms(rc))
for row,state in zip(d['rows'],meta['states']):
 t=row['step']-1;m=np.array(state['mean']);c=np.array(state['covariance']);mu=sum(float(ww)*mm for ww,mm in zip(w,m));raw=sum(float(ww)*(cc+np.outer(mm,mm)) for ww,mm,cc in zip(w,m,c));cov=raw-np.outer(mu,mu)
 assert np.allclose(mu,row['mean'],atol=1e-12,rtol=1e-13);assert np.allclose(cov,row['covariance'],atol=1e-10,rtol=1e-12)
 # Check exact saved output arithmetic independently, avoiding recombination rounding differences.
 sm=np.array(row['mean']);sc=np.array(row['covariance']);errs=metrics(sm,sc,ref['mean'][t],ref['covariance'][t]);assert errs==row['relative_rms_errors'];blocks=[metrics(sm,sc,bm[b,t],bc[b,t]) for b in range(8)];assert blocks==row['block_errors']
 baseline=old['rows'][t]['methods']['exact_h2_then_inherited_gauss']
 for key,saved in row['paired_reduction_from_collapsed'].items():
  delta=np.array([a[key]-b[key] for a,b in zip(baseline['block_errors'],blocks)]);assert float(delta.mean())==saved['block_mean_reduction'];assert float(delta.std(ddof=1)/np.sqrt(8))==saved['block_se']
assert np.allclose(d['rows'][0]['mean'],old['rows'][1]['methods']['exact_h2_then_inherited_gauss']['mean'],atol=1e-12)
out=dict(component_count=34,closure_calls=68,minimum_weight=float(w.min()),step4_errors=d['rows'][-1]['relative_rms_errors'],review_cpu=time.process_time()-start,scope='Independent raw-second recombination and all saved error/block-SE arithmetic. No closure rerun/new reference; finite diagnostic, not support preservation or final-accuracy guarantee.')
Path(__file__).with_name('sector_mixture_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
