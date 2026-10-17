import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,math,time,itertools
import numpy as np
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review');start=time.process_time()
def read(p):return json.loads(p.read_text())
protocol=read(P/'protocol.json')
for path,digest in protocol['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
base=Path(next(p for p in protocol['pins'] if p.endswith('error_captured.npz'))).parent;refs={}
for i in range(8):
 with np.load(base/f'error_batch_{i}.npz') as z:
  for ix,v,b,s in zip(z['indices'],z['positive'],z['bounds'],z['source']):refs[int(ix)]=(v,b,s)
pairs=list(itertools.combinations(range(4),2));cases=0
for name in ['controls','panel']:
 d=read(P/f'{name}.json');archive=np.load(P/('control_nodes.npz' if name=='controls' else 'panel_nodes.npz'))
 if name=='panel':assert [r['index'] for r in d['rows']]==protocol['panel_indices'] and len(d['rows'])==256 and d['complete']
 for row in d['rows']:
  label=row['label'];co=archive[label+'_cofactors'];prod=archive[label+'_products'];pc=archive[label+'_raw_pc'];om=archive[label+'_om'];assert co.shape==(6,3,128) and pc.shape==prod.shape==om.shape==(6,128)
  assert np.array_equal(prod,co[:,0]*co[:,1]);assert np.isfinite(co).all() and np.isfinite(pc).all()
  signs=np.array([(-1)**sum(set(range(4))-set(pair)) for pair in pairs])
  expected=-signs[:,None]*co[:,2]/np.sqrt(np.maximum(prod,1e-300));assert np.array_equal(pc,expected)
  diag=row['diagnostics'];checks={'negative_principal_minors':np.sum(co[:,:2]<0),'zero_principal_minors':np.sum(co[:,:2]==0),'product_floor_hits':np.sum(prod<1e-300),'denominator_floor_hits':np.sum(om<1e-300),'pc_clip_hits':np.sum(abs(pc)>1),'pc_nonfinite':np.sum(~np.isfinite(pc))}
  for key,val in checks.items():assert val==diag[key]==0
  assert row['parity'] and row['legacy_ops']==row['observed_ops']==41628 and row['difference']==row['value']-row['reference']
  if name=='panel':assert (row['reference'],row['reference_bound'],row['old_source'])==refs[row['index']]
  elif label=='orthant':assert row['reference']==1/16
  elif label.startswith('equicorrelation'):assert row['reference']==1/(5*math.comb(4,label.split('_')[1].count('m')))
  cases+=1
 archive.close()
panel=read(P/'panel.json');rows=panel['rows'];s=panel['summary'];diffs=[r['value']-r['reference'] for r in rows]
for key,val in {'max_abs_error':max(map(abs,diffs)),'L1_error':math.fsum(map(abs,diffs)),'signed_error':math.fsum(diffs),'reference_bound_sum':math.fsum(r['reference_bound'] for r in rows),'legacy_ops':sum(r['legacy_ops'] for r in rows)}.items():assert val==s[key]
exceed=sum(abs(r['difference'])>r['reference_bound'] for r in rows);maxrel=max(abs(r['difference']/r['reference']) for r in rows if r['reference'])
mp.mp.dps=80;tail=3*mp.acos(1-mp.exp(-70))/(4*mp.pi)
out=dict(control_cases=18,panel_cases=256,raw_nodes_checked=cases*6*128,max_absolute_error=s['max_abs_error'],L1_error=s['L1_error'],signed_error=s['signed_error'],reference_truncation_bound_sum=s['reference_bound_sum'],percone_reference_bound_exceedances=exceed,max_relative_difference=maxrel,raw_floor_clip_negative_minor_nonfinite_counts=0,true_omitted_tail_bound=str(tail),review_cpu=time.process_time()-start,scope='Saved arrays/source pins only. Finite panel improves markedly; source/reference differences are not enclosed by reference truncation bounds alone. No interior quadrature/floating/global mass certificate or full-cost claim.')
Path(__file__).with_name('log_cofactor_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
