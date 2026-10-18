"""Audit all saved split construction arrays and old output identities, no kernels."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,resource,time
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/positive_inner5_batch');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
g=read(P/'full_protocol.json');d=read(P/'full.json')
for p,h in g['pins'].items():assert sha(Path(p))==h
prior=read(g['saved_full']);children=read(g['child_source'])['children'];nodes=read(g['node_source'])['orders'];pilot=read(P/'outer.json')
assert d['complete'] and d['all_pass'] and len(d['rows'])==66
items={'split/'+r['fixture']+'/'+str(r['side']):r for r in children};old={(r['id'],r['order']):r for r in prior['rows'] if r['kind']=='split'}
assert len(items)==22 and len(old)==66
rules={r['n']:([.5*(float.fromhex(z)+1) for z in r['source_nodes_hex']],[.5*float.fromhex(z) for z in r['source_weights_hex']]) for r in nodes}
keys=set();files={};count=reused=0;groups={};progress=[json.loads(x) for x in (P/'full_progress.jsonl').read_text().splitlines()];savedprogress=[x['row'] for x in progress if x['stage']=='case_result'];assert savedprogress==d['rows']
for r in d['rows']:
 key=(r['id'],r['order']);assert key not in keys and key in old;keys.add(key);n=r['order'];item=items[r['id']];source=old[key]
 assert r['passed'] and all(r['parity'].values()) and all(r['construction_parity'].values())
 assert r['value_hex']==r['saved_value_hex']==source['value_hex'] and r['determinant_hex']==source['determinant_hex'] and r['branches']==source['branches'] and r['reference']==source['reference']
 path=Path(r['archive']);assert sha(path)==r['archive_sha256'];files[str(path)]=sha(path)
 if r['reused_pilot']:
  reused+=1;p=pilot['rows'][1];assert key==(p['id'],6) and path==P/'outer_case_1.npz' and r['archive_sha256']==p['archive_sha256']
 with np.load(path) as archive:a={k:archive[k] for k in archive.files}
 x,w=rules[n];raw=[[float.fromhex(v) for v in row] for row in item['rays_hex']];R=[[raw[i][j] for i in (2,3,4,0,1)] for j in range(5)]
 assert a['R'].tobytes()==np.array(R).tobytes() and a['nodes'].tobytes()==np.array(x).tobytes() and a['rule_weights'].tobytes()==np.array(w).tobytes()
 assert a['P'].shape==a['D'].shape==(n**3,5) and all(a[k].shape==(n**3,) for k in ('jacobian','weights','inner_values'))
 # Independent scalar traversal, no inner integral or determinant evaluation.
 for k in range(n**3):
  i,j,l=k//(n*n),(k//n)%n,k%n;u,v,s=x[i],x[j],x[l];aa=1.-u;bb=1.-v;cc=1.-s;ab=aa*bb;t1=aa*v;t2=ab*s;t3=ab*cc
  pp=[];dd=[]
  for row in R:
   z=u*row[0];z=z+t1*row[1];z=z+t2*row[2];z=z+t3*row[4];pp.append(z);dd.append(t3*(row[3]-row[4]))
  assert a['P'][k].tobytes()==np.array(pp).tobytes() and a['D'][k].tobytes()==np.array(dd).tobytes()
  assert float(a['jacobian'][k]).hex()==(((aa*aa)*aa)*((bb*bb)*cc)).hex() and float(a['weights'][k]).hex()==((w[i]*w[j])*w[l]).hex()
  assert math.isfinite(float(a['inner_values'][k])) and a['inner_values'][k]>0
 total=0.
 for ww,jj,vv in zip(a['weights'],a['jacobian'],a['inner_values'],strict=True):total=total+float(ww)*(float(jj)*float(vv))
 factor=float.fromhex(r['determinant_hex'])*(3./(8.*(math.pi*math.pi)));assert (factor*total).hex()==r['value_hex']
 assert sum(r['branches'].values())==r['inner_calls']==n**3;count+=n**3
 fixture=r['id'].split('/')[1];groups.setdefault((fixture,n),[]).append(r)
assert keys==set(old) and reused==1 and count==116864 and d['new_inner_calls']==count-216==116648
assert d['new_mass_evaluations']==d['new_det5_calls']==65 and d['original_pilot_extra_inner_calls']==216
assert (d['canonical_mapping_operations'],d['actually_executed_staged_mapping_operations'])==(96,114)
parentpass={}
for (name,n),rr in groups.items():
 assert len(rr)==2
 # Exact saved-output bit parity establishes unchanged previous accuracy diagnostics.
 parentpass.setdefault(n,0)
out={'pass':True,'rows':66,'parent_groups':len(groups),'construction_nodes':count,'new_inner_calls':116648,'pilot_reuses':1,'extra_original_pilot_nodes':216,'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'output_sha256':sha(P/'full.json'),'protocol_sha256':sha(P/'full_protocol.json'),'source_sha256':sha(P/'full.py'),'files':files,'scope':'All stored coordinate/Jacobian/weight bits and sequential assemblies independently reconstructed. Old mass/determinant/branch bit identities retained including lower-order losses. No inner/determinant/mass/reference/model calls. No runtime/native-cost/full-FP claim.'}
(O/'positive_inner_batch_full_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
