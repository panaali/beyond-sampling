"""Saved construction and sequential assembly check; no integral calls."""
from pathlib import Path
import hashlib,json,math,resource,time
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/positive_inner5_batch');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
g=read(P/'outer_protocol.json');d=read(P/'outer.json')
for p,h in g['pins'].items():assert sha(Path(p))==h
prior=read(g['saved_full']);orig=read(g['original_source']);child=read(g['child_source']);nodes=next(r for r in read(g['node_source'])['orders'] if r['n']==6)
x=[.5*(float.fromhex(v)+1) for v in nodes['source_nodes_hex']];w=[.5*float.fromhex(v) for v in nodes['source_weights_hex']]
items=[next(r for r in orig['fixtures'] if r['name']=='positive5orthant'),next(r for r in child['children'] if r['fixture']=='positive5orthant' and r['side']==0)]
assert d['complete'] and d['all_pass'] and len(d['rows'])==2
progress=[json.loads(x) for x in (P/'outer_progress.jsonl').read_text().splitlines()];assert len(progress)==9
files={}
for idx,(r,item) in enumerate(zip(d['rows'],items,strict=True)):
 ident=g['ids'][idx];assert r['id']==ident and r['order']==6 and r['passed'] and all(r['parity'].values()) and all(r['construction_parity'].values())
 old=next(s for s in prior['rows'] if s['id']==ident and s['order']==6)
 assert r['value_hex']==r['saved_value_hex']==old['value_hex'] and r['determinant_hex']==r['saved_determinant_hex']==old['determinant_hex'] and r['branches']==old['branches']
 path=P/f'outer_case_{idx}.npz';assert sha(path)==r['archive_sha256'];files[str(path)]=sha(path)
 with np.load(path) as archive:a={k:archive[k] for k in archive.files}
 raw=[[float.fromhex(v) for v in row] for row in item['rays_hex']];R=[[raw[i][j] for i in (2,3,4,0,1)] for j in range(5)]
 assert a['R'].tobytes()==np.array(R).tobytes() and a['nodes'].tobytes()==np.array(x).tobytes() and a['rule_weights'].tobytes()==np.array(w).tobytes()
 assert a['P'].shape==a['D'].shape==(216,5) and all(a[k].shape==(216,) for k in ('jacobian','weights','inner_values'))
 for k in range(216):
  i,j,l=k//36,(k//6)%6,k%6;u,v,s=x[i],x[j],x[l];aa=1.-u;bb=1.-v;cc=1.-s;ab=aa*bb;t1=aa*v;t2=ab*s;t3=ab*cc
  pp=[];dd=[]
  for row in R:
   z=u*row[0];z=z+t1*row[1];z=z+t2*row[2];z=z+t3*row[4];pp.append(z);dd.append(t3*(row[3]-row[4]))
  assert a['P'][k].tobytes()==np.array(pp).tobytes() and a['D'][k].tobytes()==np.array(dd).tobytes()
  assert float(a['jacobian'][k]).hex()==(((aa*aa)*aa)*((bb*bb)*cc)).hex()
  assert float(a['weights'][k]).hex()==((w[i]*w[j])*w[l]).hex()
  assert math.isfinite(float(a['inner_values'][k])) and a['inner_values'][k]>0
 total=0.
 for ww,jj,vv in zip(a['weights'],a['jacobian'],a['inner_values'],strict=True):total=total+float(ww)*(float(jj)*float(vv))
 factor=float.fromhex(r['determinant_hex'])*(3./(8.*(math.pi*math.pi)))
 assert (factor*total).hex()==r['value_hex']
 assert progress[4+4*idx]['row']==r
assert (d['new_inner_calls'],d['new_det5_calls'],d['new_mass_evaluations'],d['diagnostic_scalar_construction_nodes'])==(432,2,2,432)
out={'pass':True,'cases':2,'nodes':432,'files':files,'output_sha256':sha(P/'outer.json'),'source_sha256':sha(P/'outer.py'),'protocol_sha256':sha(P/'outer_protocol.json'),'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'scope':'Saved P/D/J/weights bit reconstruction and sequential assembly from stored positive inner values; no inner primitive/determinant/mass/model/reference calls. Finite parity only, not source-FP or native/runtime guarantee.'}
(O/'positive_inner_batch_outer_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
