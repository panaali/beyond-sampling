import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,math,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review/latest');start=time.process_time()
def read(p):return json.loads(p.read_text())
protocol=read(P/'protocol.json');assert hashlib.sha256((P/'estT_snapshot.py').read_bytes()).hexdigest()==protocol['source_sha256'];old={r['index']:r for r in read(P.parent/'panel.json')['rows']};counts=Counter();checked=0;stressfail=[]
for name in ['controls','panel']:
 d=read(P/f'{name}.json');z=np.load(P/('control_nodes.npz' if name=='controls' else 'panel_nodes.npz'))
 if name=='panel':assert d['complete'] and [r['index'] for r in d['rows']]==protocol['panel_indices']
 for r in d['rows']:
  label=r['label'];g=r['diagnostics'];assert r['parity'] and r['legacy_ops']==r['observed_ops'] and r['difference']==r['value']-r['reference'];assert g['inverse_calls']==1
  if name=='panel':assert r['reference']==old[r['index']]['reference'] and r['reference_bound']==old[r['index']]['reference_bound'];counts[g['branch']]+=1
  if g['branch']=='duffy10':
   q=z[label+'_q'];assert q.shape==(1000,) and np.isfinite(q).all() and min(q)==g['q_min']>0;assert g['scalar3det_calls']==0 and g['scalar4det_calls']==1
  else:
   Rt=z[label+'_Rt'];det=z[label+'_detRt'];angles=z[label+'_atan2_arguments'];assert Rt.shape==(128,4,4) and angles.shape==(6,2,128);assert np.sum(det<0)==g['raw_negative_detRt'] and np.sum(det==0)==g['raw_zero_detRt'];assert float(np.max(abs(np.diagonal(Rt,axis1=1,axis2=2)-1)))==g['max_unit_diagonal_drift'];assert np.sum(~np.isfinite(angles))==g['angle_nonfinite'];assert g['scalar3det_calls']==16+6*128 and g['scalar4det_calls']==129
  if r.get('kind')=='general_domain_stress':
   k=int(label.split('_')[-1]);eps=2.**(-k);truth=math.atan2(eps,-math.sqrt(1-eps*eps))/(8*math.pi);assert r['reference']==truth
   if abs(r['difference'])>1e-10:stressfail.append(label)
  else:assert abs(r['difference'])<=1e-10+r['reference_bound']
  checked+=1
 z.close()
panel=read(P/'panel.json');s=panel['summary'];rows=panel['rows'];assert counts==Counter(duffy10=117,log128_atan2=139);assert len(stressfail)==4 and read(P/'controls.json')['stress_gate_pass'] is False
for key,val in {'max_abs_error':max(abs(r['difference']) for r in rows),'L1_error':math.fsum(abs(r['difference']) for r in rows),'signed_error':math.fsum(r['difference'] for r in rows),'legacy_ops':sum(r['legacy_ops'] for r in rows),'reference_only_bound_exceedances':sum(abs(r['difference'])>r['reference_bound'] for r in rows)}.items():assert val==s[key]
out=dict(cases=checked,actual_panel=256,branches=dict(counts),generic_stress_failures=stressfail,max_absolute_error=s['max_abs_error'],L1_error=s['L1_error'],reference_only_bound_exceedances=s['reference_only_bound_exceedances'],max_diagonal_drift=s['max_unit_diagonal_drift'],review_cpu=time.process_time()-start,scope='Pinned latest-source saved nodes/references and analytic stress formulas. Actual fixed panel passes preset absolute diagnostic gate; four generic unit-ray stress cases fail. No universal bound, full floating certificate or full-cost inference; no quadrature rerun.')
Path(__file__).with_name('latest_mass_panel_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
