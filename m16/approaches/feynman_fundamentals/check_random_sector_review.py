import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,hashlib,math,time,numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/structured_prefix_mixture');start=time.process_time();proto=json.loads((P/'random_protocol.json').read_text());summary=json.loads((P/'random_summary.json').read_text());rows=[];outputs=0;bad=[];better=0
for h in proto['widths']:
 for dim in (1,h):
  group=[]
  for seed in proto['model_seeds']:
   stem=f'random_H{h}_d{dim}_s{seed}';d=json.loads((P/(stem+'.json')).read_text());path=P/(stem+'.npz');assert d['complete'] and hashlib.sha256(path.read_bytes()).hexdigest()==d['archive_sha256'];a=np.load(path);rng=np.random.default_rng(seed);wi=rng.standard_normal((h,dim))*math.sqrt(2/dim);wh=rng.standard_normal((h,h))*math.sqrt(2/h);wo=rng.standard_normal((h,h))*math.sqrt(1/h)
   assert all(np.array_equal(x,a[k]) for x,k in [(wi,'wi'),(wh,'wh'),(wo,'wo')]);assert hashlib.sha256(wi.tobytes()+wh.tobytes()+wo.tobytes()).hexdigest()==d['weight_sha256'];assert d['reference_seed']==202610170690+1000*h+10*dim+seed
   bm=a['reference_block_mean'];bs=a['reference_block_raw_second'];mean=bm.mean(0);var=bs.mean(0)-mean**2;bv=bs-bm**2;assert np.array_equal(mean,a['reference_mean']) and np.array_equal(var,a['reference_variance'])
   for name,r in d['methods'].items():
    out=wo@np.array(r['hidden_mean']);assert np.allclose(out,r['output_mean'],atol=1e-14,rtol=1e-13);out=np.array(r['output_mean']);e=(out-mean)**2/var;block=np.mean((out-bm)**2/bv,axis=1);assert np.array_equal(e,r['per_output_vMSE']);assert float(e.mean())==r['mean_vMSE'];assert np.array_equal(block,r['block_vMSE']);assert r['floor_subtracted_vMSE']==r['mean_vMSE']-1/proto['reference_samples'];outputs+=1
   if dim==1:
    w=a['component_weights'];m=a['component_means'];c=a['component_covariances'];mu=np.sum(w[:,None]*m,axis=0);cov=np.sum(w[:,None,None]*(c+m[:,:,None]*m[:,None,:]),axis=0)-np.outer(mu,mu);r=d['methods']['sector_mixture'];assert np.allclose(mu,r['hidden_mean'],atol=1e-13,rtol=1e-12) and np.allclose(cov,r['hidden_covariance'],atol=1e-12,rtol=1e-11);better+=r['mean_vMSE']<d['methods']['gauss']['mean_vMSE']
    if r['mean_vMSE']>d['methods']['collapsed_h2']['mean_vMSE']:bad.append(dict(H=h,seed=seed))
   group.append(d);rows.append(d)
  saved=next(r for r in summary['rows'] if r['H']==h and r['d']==dim)
  for name,v in saved['mean_vMSE'].items():assert float(np.mean([r['methods'][name]['mean_vMSE'] for r in group]))==v
assert len(rows)==24 and outputs==72 and better==summary['scalar_mixture_better_than_gauss']==12 and bad==summary['scalar_mixture_worse_than_collapsed'];assert math.isclose(sum(r['cpu_seconds'] for r in rows),summary['cell_cpu_seconds'],rel_tol=1e-14)
out=dict(cells=24,method_outputs=72,scalar_mixture_better_than_gaussian=better,scalar_mixture_worse_than_collapsed=bad,all_weight_bytes_reconstructed=True,review_cpu=time.process_time()-start,scope='Saved reference/mixture arithmetic, fixed three seeds per width. No new inputs or closure run. 1/N is a nominal reference-noise scale using estimated variance, not exact debiasing/confidence certificate; no monotone/asymptotic or equal-cost inference.')
Path(__file__).with_name('random_sector_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
