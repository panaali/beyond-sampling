import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from decimal import Decimal as D,localcontext
from collections import Counter
import json,hashlib,math,time,itertools
import numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review');start=time.process_time()
def read(p):return json.loads(p.read_text())
def near(a,b):assert math.isclose(float(a),b,rel_tol=2e-14,abs_tol=1e-24),(a,b)
s=read(P/'full_summary.json')
for name,digest in s['pins'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
protocol=read(P/'protocol.json');base=Path(next(p for p in protocol['pins'] if p.endswith('error_captured.npz'))).parent;panel={r['index']:r for r in read(P/'panel.json')['rows']};panelnodes=np.load(P/'panel_nodes.npz');allrows=[];counts=Counter();used=[];bad=[];clipped=[];negative=[];rawchecked=0;processcpu=0.
for batch in range(8):
 d=read(P/f'full_{batch}.json');assert d['complete'];z=np.load(base/f'error_batch_{batch}.npz');assert [r['index'] for r in d['rows']]==z['indices'].tolist();raw=np.load(P/f'full_nodes_{batch}.npz');ids=z['indices'].tolist();sentinel={ids[0],ids[len(ids)//2],ids[-1]};processcpu+=d['cpu']
 for r,p,b,old in zip(d['rows'],z['positive'],z['bounds'],z['source']):
  idx=r['index'];assert r['reference']==p and r['reference_bound']==b and r['old_source']==old and r['difference']==r['value']-p;assert r['parity'] and r['legacy_ops']==r['observed_ops'];diag=r['diagnostics']
  if r['reused_from_panel']:
   used.append(idx);assert all(r[k]==v for k,v in panel[idx].items());nodes=panelnodes
  else:nodes=raw
  anomaly=r['value']<0 or abs(r['difference'])>1e-10+b or diag['cutoff'] or any(diag.get(k,0) for k in ['negative_principal_minors','zero_principal_minors','product_floor_hits','denominator_floor_hits','pc_clip_hits','pc_nonfinite'])
  if abs(r['difference'])>1e-10+b:bad.append(idx)
  if r['value']<0:negative.append(idx)
  if diag.get('pc_clip_hits',0):clipped.append(idx)
  counts.update({k:diag.get(k,0) for k in ['negative_principal_minors','zero_principal_minors','product_floor_hits','denominator_floor_hits','pc_clip_hits','pc_nonfinite','scalar_3det_calls']});counts['inverse_calls']+=diag['inverse_calls'];counts['scalar_4det_calls']+=sum(shape==[4,4] for shape in diag['determinant_shapes']) if diag['cutoff'] else diag['scalar_4det_calls']
  if not diag['cutoff'] and (anomaly or idx in sentinel or r['reused_from_panel']):
   co=nodes[f'{idx}_cofactors'];prod=nodes[f'{idx}_products'];pc=nodes[f'{idx}_raw_pc'];om=nodes[f'{idx}_om'];assert np.array_equal(prod,co[:,0]*co[:,1]);signs=np.array([(-1)**sum(set(range(4))-set(pair)) for pair in itertools.combinations(range(4),2)]);assert np.array_equal(pc,-signs[:,None]*co[:,2]/np.sqrt(np.maximum(prod,1e-300)));assert np.sum(abs(pc)>1)==diag['pc_clip_hits'];assert np.sum(co[:,:2]<0)==diag['negative_principal_minors'];assert np.sum(prod<1e-300)==diag['product_floor_hits'];rawchecked+=1
  allrows.append(r)
 raw.close();z.close()
assert sorted(r['index'] for r in allrows)==list(range(35098));assert len(used)==len(set(used))==256 and set(used)==set(panel)
for k,v in counts.items():assert v==s[k],(k,v,s[k])
assert bad==[31514] and negative==[4885] and len(clipped)==9 and counts['pc_clip_hits']==108 and s['finite_gate_pass'] is False
with localcontext() as ctx:
 ctx.prec=65;mass=sum(D(r['value']) for r in allrows);pos=sum(D(r['reference']) for r in allrows);diff=[D(r['value'])-D(r['reference']) for r in allrows];near(mass,s['source_mass']);near(pos,s['positive_mass']);near(1-mass,s['source_accuracy']);near(sum(diff),s['signed_error']);near(sum(abs(v) for v in diff),s['L1_error']);near(max(abs(v) for v in diff),s['max_abs_error'])
assert sum(r['legacy_ops'] for r in allrows)==s['legacy_ops_once'];near(math.fsum(r['source_cpu'] for r in allrows),s['source_cpu_once']);near(math.fsum(r['diagnostic_cpu'] for r in allrows),s['observer_cpu_once']);near(processcpu,s['shard_process_cpu']);assert sum(abs(r['difference'])>r['reference_bound'] for r in allrows)==s['reference_only_bound_exceedances'];assert sum(r['diagnostics']['cutoff'] for r in allrows)==73
out=dict(cases=35098,reused256_once=True,raw_node_cones_checked=rawchecked,gate_failure_indices=bad,negative_indices=negative,clipped_indices=clipped,clipped_nodes=108,source_accuracy=s['source_accuracy'],positive_accuracy=s['positive_accuracy'],L1_error=s['L1_error'],max_absolute_error=s['max_abs_error'],max_relative_difference=s['max_relative_error'],source_numeric_proxy=s['legacy_ops_once'],det4_calls=counts['scalar_4det_calls'],det3_calls=counts['scalar_3det_calls'],review_cpu=time.process_time()-start,scope='Every saved index/reference/aggregate plus all anomaly/sentinel and reused raw nodes. No quadrature rerun. Finite preset absolute gate fails one cone; one negative mass and clipping remain despite excellent aggregate agreement. Conditional reference bounds and full-cost limitations persist.')
Path(__file__).with_name('log_cofactor_full_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
