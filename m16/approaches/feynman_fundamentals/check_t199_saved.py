"""Second-reviewer exact scalar contraction of fixed T194 evidence."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
import hashlib,json,math,time
P=Path('/workspace/work/recover-small_h2_910/review/width16_evidence_review');O=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f(x):return F(int(x[0]),int(x[1]))
start=time.process_time();r=load(P/'results.json');proto=load(P/'protocol.json');manifest=load(P/'manifest.json')
assert r['complete'] and r['protocol_sha256']==sha(P/'protocol.json') and r['source_sha256']==sha(P/'audit.py')
for p,h in proto['pins'].items():assert sha(Path(p))==h,p
assert len(manifest['files'])==82
for p in manifest['files']:assert sha(P/p['snapshot_path'])==p['sha256']
groups=defaultdict(list);fallback=0
for a in r['rows']:
 raw={}
 for mode,b in a['source_paths'].items():
  assert sha(P/b['path'])==b['sha256'];v=load(P/b['path'])
  if b['source']=='rescore_fallback':fallback+=1;v=v[mode]
  raw[mode]=v
 ex,rc=raw['exact'],raw['recipe108'];p,q=F(ex['acc']),F(rc['acc']);H,T=a['H'],a['T'];cost=F(6000*2*T*H*(H+2),5760)
 assert f(a['exact_accuracy'])==p and f(a['recipe_accuracy'])==q and f(a['error'])==q-p
 ratio=(q-p)**2*rc['ops']/(cost*p*(1-p));cs=2*T*H*(H+2)+T*H+3*T-2
 assert f(a['published_ratio'])==ratio and a['canonical_sampler']==cs and f(a['conditional_T196_2ops_ratio'])==ratio*2*cost/cs
 assert a['recipe_cells']==rc['cells'] and a['exact_cells']==ex['cells']
 assert f(a['recipe_cells_per_pattern'])==F(rc['cells'],a['floating_leaf_patterns']) and f(a['recipe_ops_per_exact_ops'])==F(rc['ops'],ex['ops'])
 assert a['floating_leaf_patterns']==(ex['regions_exact'] or a['sampled_patterns_2M'])
 for m in a['reference_metadata']:
  assert m['not_literal_saved_hits'] and abs(F(m['proportion'])-F(m['inferred_hits'],m['n']))<=F(math.ulp(m['proportion']))
 groups[(H,T)].append(a)
assert len(r['rows'])==32 and fallback==9 and len(groups)==4
for g in r['groups']:
 rs=groups[(g['H'],g['T'])];assert g['models']==len(rs)
 for k,fn,field in [('min_cells_per_pattern',min,'recipe_cells_per_pattern'),('max_cells_per_pattern',max,'recipe_cells_per_pattern'),('max_original_ratio',max,'published_ratio'),('max_conditional_T196_ratio',max,'conditional_T196_2ops_ratio'),('min_recipe_ops_per_exact_ops',min,'recipe_ops_per_exact_ops'),('max_recipe_ops_per_exact_ops',max,'recipe_ops_per_exact_ops')]:assert f(g[k])==fn(f(a[field]) for a in rs)
assert len(r['controls'])==4
for c in r['controls']:
 R=[[f(v) for v in row] for row in c['rays']];m=[sum(row[j] for row in R)/2 for j in range(2)];fm=[(float(R[0][j])+float(R[1][j]))/2 for j in range(2)]
 assert list(map(f,c['exact_mean']))==m and f(c['exact_margin'])==m[0]-m[1] and [x.hex() for x in fm]==c['float_mean']
 assert c['float_margin']==fm[0]-fm[1] and c['exact_gate']==(m[0]>m[1]) and c['float_gate']==(fm[0]>fm[1])
 assert f(c['determinant'])==R[0][0]*R[1][1]-R[0][1]*R[1][0]!=0
out={'complete':True,'owner_commit':'98b06227','result_sha256':sha(P/'results.json'),'protocol_sha256':sha(P/'protocol.json'),'snapshot_files':82,'rows':32,'fallback_records':9,'synthetic_center_controls':4,'groups':r['groups'],'review_cpu':time.process_time()-start,'scope':'Saved scalar ratios/count semantics and four dyadic rounded-center examples. Floating leaf-pattern counts are not certified exact arithmetic region counts or minimal affine-map counts. No trained counterexample, real model replay, complete-cost bound or independent mass truth. Author original history/folder remains separate from byte-pinned review snapshot.'}
(O/'t199_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'rows':32,'fallback':9,'review_cpu':out['review_cpu']}))
