"""Independent k4 COUNT review: graph-plane arrangement plus strict witnesses.
No author clipping/forward calls; full 4D ray completeness is deliberately not claimed.
"""
from check_t219_controls_saved import history_forms,primitive,dot,sha
from pathlib import Path
from itertools import combinations
from collections import Counter
import gzip,json,time,traceback
P=Path('/workspace/work/recover-small_h2_34/review/target_prefix_recount');H=Path(__file__).resolve().parent;OUT=H/'t219_k4_review_progress.jsonl'
def read(p):return json.loads(Path(p).read_text())
def readgz(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def normalize(v):
 v=primitive(v)
 return tuple(-x for x in v) if next(x for x in v if x)<0 else v
def exact_rank(rows,d):
 basis=[]
 for row in rows:
  v=list(row)
  for p,b in basis:
   if v[p]:
    c=v[p];v=[b[p]*x-c*y for x,y in zip(v,b)]
    if any(v):v=list(primitive(v))
  if any(v):
   p=next(j for j,x in enumerate(v) if x);basis.append((p,tuple(v)))
   if len(basis)==d:return d
 return len(basis)
def arrangement(A,parent,sign):
 # Every graph is defined over the entire parent polygon. Adding a graph adds
 # as many regions as its intersections with earlier graphs cut on that polygon.
 planes=sorted({tuple(x if a[-1]>0 else -x for x in primitive(a)) for a in A})
 regions=1;increments=[];linecount=0;pointcount=0
 for k,a in enumerate(planes):
  lines=set()
  for b in planes[:k]:
   v=tuple(a[q]*b[3]-b[q]*a[3] for q in range(3))
   assert any(v)
   signs=[dot(v,r) for r in parent['rays']]
   if min(signs)<0<max(signs):lines.add(normalize((sign*v[0],v[1],v[2])))
  ll=sorted(lines);points={}
  for i,j in combinations(range(len(ll)),2):
   c,d=ll[i],ll[j];den=c[1]*d[2]-c[2]*d[1]
   if den==0:continue
   yn=c[2]*d[0]-c[0]*d[2];zn=c[0]*d[1]-c[1]*d[0];sgn=1 if den>0 else -1
   xyz=(sign*abs(den),sgn*yn,sgn*zn)
   if not all(dot(n,xyz)>0 for n in parent['normals']):continue
   point=primitive((abs(den),sgn*yn,sgn*zn));points.setdefault(point,set()).update((i,j))
  inc=1+len(ll)+sum(len(v)-1 for v in points.values());regions+=inc;increments.append(inc);linecount+=len(ll);pointcount+=len(points)
 return dict(count=regions,distinct_planes=len(planes),increments=increments,slice_lines=linecount,slice_interior_points=pointcount)
def main():
 assert not OUT.exists(),'preserve existing reviewer attempt'
 start=time.process_time()
 def event(v):
  with OUT.open('a') as f:f.write(json.dumps(v,sort_keys=True)+'\n');f.flush()
 event(dict(stage='started'))
 # Finite algebra identities guard parallel, crossing, concurrent and boundary cases.
 box=dict(rays=[[1,x,y] for x in (-1,1) for y in (-1,1)],normals=[[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])
 controls=[([[0,0,0,1]],2),([[0,0,0,1],[-1,0,0,1]],3),([[0,0,0,1],[0,-1,0,1],[0,0,-1,1]],8),([[0,0,0,1],[0,-1,0,1],[0,-2,0,1]],6),([[0,0,0,1],[-1,-1,0,1]],3)]
 for a,n in controls:assert arrangement(a,box,1)['count']==n
 r=read(P/'depth4/summary.json');p=read(P/'depth4_protocol.json');g=read(P/'depth4_gate.json');prior=read(P/'depth3/summary.json');old=read(P/'target_k2_result.json')
 assert r['status']=='complete' and r['gate']==g and g['clearance'] and r['protocol_sha256']==g['protocol_sha256']==sha(P/'depth4_protocol.json')
 for name,h in {**p['pins'],**g['pins']}.items():assert sha(name)==h
 rev=g['k3_saved_review'];assert sha(rev['path'])==rev['sha256'] and read(rev['path'])['pass_all']
 assert r['weights_input_sha256']==sha(P/'target_k2_result.json') and r['parent_summary_sha256']==sha(P/'depth3/summary.json')
 parents=[];locations=[]
 for item in prior['parents']:
  f=P/'depth3'/item['path'];assert sha(f)==item['sha256'];d=readgz(f)
  for i,s in enumerate(d['children']):parents.append(s);locations.append(dict(archive=item['path'],child_index=i))
 assert len(parents)==len(r['parents'])==792
 case=old['weights'];wi=case['wi'];W=case['wh'];D=case['D'];assert all(wi)
 seen=set();counts=Counter();reports=[];peak=0;cpu=0.;rays=0;normals=0
 ee=[dict(phase='attempt',stage='load_saved_parent_records'),dict(phase='result',stage='load_saved_parent_records',count=792)]
 for idx,(parent,item) in enumerate(zip(parents,r['parents'],strict=True)):
  assert time.process_time()-start<100,'reviewer100CPU cap; preserve completed parent checks'
  assert item['index']==idx and item['path']==f'parent_{idx:03d}.json.gz';f=P/'depth4'/item['path'];assert sha(f)==item['sha256'];d=readgz(f)
  assert d['status']==item['status']=='complete' and d['parent_index']==idx and d['parent_location']==locations[idx] and d['parent_history']==parent['history']
  assert d['counts']==item['counts'] and d['max_retained_bits']==item['max_retained_bits'] and d['cpu_seconds']==item['cpu_seconds']
  assert len(d['children'])==len(d['certificates'])==item['children']==item['certificates']
  counts.update(d['counts']);peak=max(peak,d['max_retained_bits']);cpu+=d['cpu_seconds']
  ph=tuple(map(tuple,parent['history']));hidden,pforms,zero=history_forms(case,ph);assert hidden==parent['hidden'];oldforms=[tuple(v)+(0,) for v in pforms]
  sign=1 if parent['rays'][0][0]>0 else -1
  # Rays on x0=0 may sort first: obtain half sign from the first-layer mask instead.
  signs=[s for s in (-1,1) if tuple(int(s*w>0) for w in wi)==ph[0]];assert len(signs)==1;sign=signs[0]
  A=[[sum(W[i][j]*hidden[j][q] for j in range(16)) for q in range(3)]+[wi[i]*D**3] for i in range(16)]
  comp=arrangement(A,parent,sign);assert comp['count']==len(d['children'])
  local=set()
  for s,c in zip(d['children'],d['certificates'],strict=True):
   hist=tuple(map(tuple,s['history']));assert len(hist)==4 and hist[:3]==ph and hist not in seen;seen.add(hist);local.add(hist)
   assert len(hist[-1])==16 and all(b in (0,1) for b in hist[-1])
   expected=[a if on else [0]*4 for a,on in zip(A,hist[-1])];assert s['hidden']==expected
   forms=oldforms+[tuple(x if on else -x for x in a) for a,on in zip(A,hist[-1])]
   assert c['zero_forms']==zero and c['strict_forms']==len(forms)
   rr=s['rays'];nn=s['normals'];assert len(rr)==len(set(map(tuple,rr))) and len(nn)==len(set(map(tuple,nn)))
   assert all(len(v)==4 and tuple(v)==primitive(v) for v in rr+nn)
   witness=primitive(tuple(sum(v[q] for v in rr) for q in range(4)));assert list(witness)==c['witness'] and sign*witness[0]>0
   assert all(dot(v,witness)>0 for v in forms)
   assert set(map(tuple,nn))<={primitive(v) for v in forms}
   assert all(dot(n,v)>=0 for n in forms for v in rr)
   assert exact_rank(rr,4)==4 and exact_rank(nn,4)==4
   # Feasibility and individual extremality are checked; missing extra rays are
   # not excluded here and are unnecessary for the separately proved COUNT.
   for v in rr:assert exact_rank([n for n in nn if dot(n,v)==0],4)==3
   assert all(dot(n,witness)>0 for n in nn)
   rays+=len(rr);normals+=len(nn)
  assert len(local)==comp['count'];report=dict(parent=idx,children=len(local),arrangement=comp,payload_sha256=sha(f));reports.append(report)
  event(dict(stage='parent_checked',**report));ee.extend([dict(phase='attempt',parent=idx),dict(phase='result',parent=idx,children=len(local),status='complete')])
 assert len(seen)==sum(v['children'] for v in reports)==r['count']==r['author_reported_k4']==21032 and r['author_count_agrees']
 ee.append(dict(phase='complete',count=21032,author_count_agrees=True));assert [json.loads(x) for x in (P/'depth4/progress.jsonl').read_text().splitlines()]==ee
 assert dict(counts)==r['counts'] and peak==r['max_retained_bits'] and cpu<=r['cpu_seconds']
 assert all(r[k]==0 for k in ['framework_forward_calls','checkpoint_decodes','mass_calls','random_calls'])
 out=dict(complete=True,pass_all=True,raw_commit='7c7b7c98',pins={n:sha(P/n) for n in ['depth4/summary.json','depth4/progress.jsonl','depth4_protocol.json','depth4_gate.json','depth3/summary.json','target_k2_result.json']},checker_sha256=sha(__file__),helper_sha256=sha(H/'check_t219_controls_saved.py'),parents=792,exact_gate_histories=21032,ray_records=rays,normal_records=normals,author_progress_events=len(ee),reviewer_progress_sha256=sha(OUT),arrangement_control_cases=5,parent_count_range=[min(x['children'] for x in reports),max(x['children'] for x in reports)],reported_ledger_sum=dict(counts),reported_max_bits=peak,author_cpu=r['cpu_seconds'],review_cpu=time.process_time()-start,new_author_geometry_model_calls=0,scope='Every distinct exact strict-history witness gives a nonempty full-dimensional gate region; independent sequential graph-plane arrangements over each previously audited parent polygon give the exhaustive count. Their equality proves21032 exact stored-weight gate histories at k4. All saved maps/rays checked for binding, feasibility, rank and individual extremality, but full4D ray-list completeness NOT independently certified. Thus no k5 reuse approval. Named counts aggregated only, no operation/peak-bit replay, Gaussian mass/minimal affine-region/T10 cost or factor30-growth conclusion.')
 (H/'t219_k4_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 try:main()
 except BaseException:
  (H/'t219_k4_reviewer_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2)+'\n');raise
