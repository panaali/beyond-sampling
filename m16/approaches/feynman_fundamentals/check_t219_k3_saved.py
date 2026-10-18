"""Independent saved k3 certificates plus strip-line arrangement completeness.
No author imports, clipping/extrusion, checkpoint decode or network forwards.
"""
from check_t219_controls_saved import cone_check,history_forms,primitive,dot,sha
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from collections import Counter
import gzip,json,time
P=Path('/workspace/work/recover-small_h2_34/review/target_prefix_recount');H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def main():
 start=time.process_time();r=read(P/'depth3/summary.json');p=read(P/'depth3_protocol.json');g=read(P/'depth3_gate.json');old=read(P/'target_k2_result.json')
 assert r['status']=='complete' and r['gate']==g and g['clearance'] and r['protocol_sha256']==g['protocol_sha256']==sha(P/'depth3_protocol.json')
 for name,h in {**p['pins'],**g['pins']}.items():assert sha(name)==h
 review=g['k2_saved_review'];assert sha(review['path'])==review['sha256'] and read(review['path'])['pass_all']
 assert r['weights_input_sha256']==sha(P/'target_k2_result.json')
 parents=old['cone_levels'][1]['states'];assert len(parents)==len(r['parents'])==34
 case=old['weights'];wi=case['wi'];W=case['wh'];D=case['D'];assert len(wi)==16 and all(wi)
 planar={tuple(map(tuple,row['history'])):(half['sign'],row['bounds']) for half in old['planar'] for row in half['rows']}
 expected_events=[dict(phase='attempt',stage='load_saved_parent_records'),dict(phase='result',stage='load_saved_parent_records',count=34)]
 seen=set();counts=Counter();reports=[];peak=0;cpu=0.;raycount=0;normalcount=0
 for idx,(parent,record) in enumerate(zip(parents,r['parents'],strict=True)):
  assert record['index']==idx and record['path']==f'parent_{idx:02d}.json.gz'
  f=P/'depth3'/record['path'];assert sha(f)==record['sha256'];d=json.loads(gzip.decompress(f.read_bytes()))
  assert d['status']==record['status']=='complete' and d['parent_index']==idx and d['parent_history']==parent['history']
  assert len(d['children'])==len(d['certificates'])==record['children']==record['certificates']
  assert d['counts']==record['counts'] and d['cpu_seconds']==record['cpu_seconds'] and d['max_retained_bits']==record['max_retained_bits']
  assert all(type(v) is int and v>=0 for v in d['counts'].values());counts.update(d['counts']);peak=max(peak,d['max_retained_bits']);cpu+=d['cpu_seconds']
  parenthist=tuple(map(tuple,parent['history']));sign,bounds=planar[parenthist];lo,hi=[None if q is None else F(*q) for q in bounds]
  # All complete third-step boundary lines have nonzero fresh coefficient.
  A=[[sum(W[i][j]*parent['hidden'][j][q] for j in range(16)) for q in range(2)]+[wi[i]*D*D] for i in range(16)]
  lines=sorted({(F(-a[1],a[2]),F(-sign*a[0],a[2])) for a in A})
  intersections={}
  for i,j in combinations(range(len(lines)),2):
   m,b=lines[i];n,c=lines[j]
   if m==n:continue
   y=(c-b)/(m-n)
   if (lo is not None and y<=lo) or (hi is not None and y>=hi):continue
   z=m*y+b;intersections.setdefault((y,z),set()).update((i,j))
  expected=len(lines)+1+sum(len(v)-1 for v in intersections.values())
  assert expected==len(d['children'])
  histories=set()
  for s,cert in zip(d['children'],d['certificates'],strict=True):
   hist=tuple(map(tuple,s['history']));assert len(hist)==3 and hist[:2]==parenthist and hist not in seen;seen.add(hist);histories.add(hist)
   w=cone_check(s,cert['witness']);hidden,forms,zero=history_forms(case,hist)
   assert hidden==s['hidden'] and cert['zero_forms']==zero and cert['strict_forms']==len(forms)
   assert all(dot(v,w)>0 for v in forms)
   # Saved H-system is a subset of history inequalities. Its complete extreme
   # rays satisfy every history inequality, proving equality of the two cones.
   canonical={primitive(v) for v in forms}
   assert set(map(tuple,s['normals']))<=canonical
   assert all(dot(n,v)>=0 for n in forms for v in s['rays'])
   assert w[0]*sign>0
   y=F(w[1],sign*w[0]);z=F(w[2],sign*w[0])
   assert (lo is None or lo<y) and (hi is None or y<hi)
   assert tuple(int(sign*a[0]+a[1]*y+a[2]*z>0) for a in A)==hist[-1]
   raycount+=len(s['rays']);normalcount+=len(s['normals'])
  assert len(histories)==expected
  reports.append(dict(parent=idx,children=expected,distinct_lines=len(lines),interior_intersections=len(intersections),intersection_multiplicities=dict(Counter(len(v) for v in intersections.values())),child_payload_sha256=sha(f)))
  expected_events.extend([dict(phase='attempt',parent=idx),dict(phase='result',parent=idx,children=expected,status='complete')])
 assert len(seen)==sum(v['children'] for v in reports)==r['count']==r['author_reported_k3']==792 and r['author_count_agrees']
 expected_events.append(dict(phase='complete',count=792,author_count_agrees=True))
 assert [json.loads(x) for x in (P/'depth3/progress.jsonl').read_text().splitlines()]==expected_events
 assert dict(counts)==r['counts'] and peak==r['max_retained_bits'] and cpu<=r['cpu_seconds']
 assert all(r[k]==0 for k in ['framework_forward_calls','checkpoint_decodes','mass_calls','random_calls'])
 out=dict(complete=True,pass_all=True,raw_commit='87536b5f',pins={n:sha(P/n) for n in ['depth3/summary.json','depth3/progress.jsonl','depth3_protocol.json','depth3_gate.json','target_k2_result.json','K3_COVERAGE_PLAN.md']},checker_sha256=sha(__file__),helper_sha256=sha(H/'check_t219_controls_saved.py'),parents=34,exact_gate_histories=792,ray_records=raycount,normal_records=normalcount,progress_events=len(expected_events),parent_completeness=reports,reported_ledger_sum=dict(counts),reported_max_bits=peak,author_cpu=r['cpu_seconds'],review_cpu=time.process_time()-start,new_author_geometry_model_calls=0,scope='Every saved child H-cone checked by independent active-normal minor enumeration, extreme rays/rank/strict witness, closed-column state maps and history-form equivalence. Independent complete affine-line arrangements on each parent strip give each child count, including parallel/coincident/concurrent and boundary rules. Thus792 full-dimensional exact stored-weight gate histories through k3 are verified, not minimal affine maps, k4/k5/T10 growth, Gaussian masses or costs. Internal operation/peak-bit calls are not replayed; saved ledgers are aggregated only.')
 (H/'t219_k3_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='parent_completeness'}))
if __name__=='__main__':main()
