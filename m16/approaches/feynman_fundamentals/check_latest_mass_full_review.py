import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from decimal import Decimal as D,localcontext
from collections import Counter
import json,hashlib,math,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review/latest');start=time.process_time()
def read(p):return json.loads(p.read_text())
def near(a,b):assert math.isclose(float(a),b,rel_tol=2e-14,abs_tol=1e-24),(a,b)
s=read(P/'full_summary.json')
for name,h in s['pins'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
pins=read(P.parent/'protocol.json')['pins'];base=Path(next(p for p in pins if p.endswith('error_captured.npz'))).parent
saved={r['index']:r for r in read(P/'panel.json')['rows']};pn=np.load(P/'panel_nodes.npz');cn=np.load(P/'control_nodes.npz')
for r in read(P/'controls.json')['rows']:
 if r['kind']=='actual_anomaly':
  idx=int(r['label'].split('_')[-1]);assert idx not in saved;saved[idx]=dict(r,index=idx)
assert len(saved)==266
rows=[];used=[];counts=Counter();rawchecked=0;shardcpu=0
for batch in range(8):
 d=read(P/f'full_{batch}.json');assert d['complete'];z=np.load(base/f'error_batch_{batch}.npz');raw=np.load(P/f'full_nodes_{batch}.npz');ids=z['indices'].tolist();assert ids==[r['index'] for r in d['rows']];sentinel={ids[0],ids[len(ids)//2],ids[-1]};shardcpu+=d['cpu']
 for r,p,b,old in zip(d['rows'],z['positive'],z['bounds'],z['source']):
  idx=r['index'];g=r['diagnostics'];assert r['reference']==p and r['reference_bound']==b and r['original_source']==old;assert r['difference']==r['value']-p;assert r['parity'] and r['legacy_ops']==r['observed_ops'];assert math.isfinite(r['value']) and r['value']>=0 and abs(r['difference'])<=1e-10+b
  if r['reused']:
   used.append(idx);assert all(r[k]==v for k,v in saved[idx].items());nodes=cn if r.get('kind')=='actual_anomaly' else pn
  else:nodes=raw
  thin=g['branch']=='duffy10';assert g['branch'] in ('duffy10','log128_atan2');counts['thin' if thin else 'wide']+=1
  assert g['inverse_calls']==1 and g['scalar3det_calls']==(0 if thin else 784) and g['scalar4det_calls']==(1 if thin else 129)
  for k in ['raw_negative_detRt','raw_zero_detRt','angle_nonfinite','denominator_floor_hits']:assert g.get(k,0)==0
  for k in ['inverse_calls','scalar3det_calls','scalar4det_calls']:counts[k]+=g[k]
  if r['reused'] or idx in sentinel:
   label=r['label']
   if thin:
    q=nodes[label+'_q'];assert q.shape==(1000,) and np.isfinite(q).all() and min(q)==g['q_min']>0
   else:
    Rt=nodes[label+'_Rt'];det=nodes[label+'_detRt'];ang=nodes[label+'_atan2_arguments'];assert Rt.shape==(128,4,4) and det.shape==(128,) and ang.shape==(6,2,128);assert np.isfinite(Rt).all() and np.isfinite(det).all() and np.isfinite(ang).all();assert (det>0).all() and min(det)==g['min_detRt'];assert np.max(abs(np.diagonal(Rt,axis1=1,axis2=2)-1))==g['max_unit_diagonal_drift']
   rawchecked+=1
  rows.append(r)
 raw.close();z.close()
assert sorted(r['index'] for r in rows)==list(range(35098));assert len(used)==len(set(used))==266 and set(used)==set(saved)
for k,v in counts.items():assert v==s[k],(k,v,s[k])
with localcontext() as ctx:
 ctx.prec=65;mass=sum(D(r['value']) for r in rows);pos=sum(D(r['reference']) for r in rows);diff=[D(r['value'])-D(r['reference']) for r in rows];near(mass,s['source_error_mass']);near(pos,s['positive_error_mass']);near(1-mass,s['source_accuracy']);near(sum(diff),s['signed_error']);near(sum(abs(v) for v in diff),s['L1_error']);near(max(abs(v) for v in diff),s['max_abs_error'])
assert sum(r['legacy_ops'] for r in rows)==s['legacy_ops_once'];near(math.fsum(r['source_cpu'] for r in rows),s['source_cpu_once']);near(math.fsum(r['diagnostic_cpu'] for r in rows),s['observer_cpu_once']);near(shardcpu,s['shard_cpu']);assert sum(abs(r['difference'])>r['reference_bound'] for r in rows)==s['reference_only_bound_exceedances'];near(math.fsum(r['reference_bound'] for r in rows),s['reference_bound_sum']);assert s['finite_gate_pass'] and read(P/'controls.json')['stress_gate_pass'] is False
out=dict(cases=len(rows),reused_once=len(used),raw_node_cones_checked=rawchecked,counts=dict(counts),source_accuracy=s['source_accuracy'],positive_accuracy=s['positive_accuracy'],max_absolute_difference=s['max_abs_error'],L1_difference=s['L1_error'],reference_only_bound_exceedances=s['reference_only_bound_exceedances'],source_proxy=s['legacy_ops_once'],review_cpu=time.process_time()-start,scope='All original indices/references/reused records, source pins, Decimal aggregates and saved sentinel/reused nodes checked without candidate or reference integrations. Finite absolute gate passes; generic thin stress failures persist. Reference truncation bounds do not enclose candidate floating errors; no universal or complete-cost certificate.')
Path(__file__).with_name('latest_mass_full_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
