"""Independent exact saved table audit; no author primitive imports or calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
B=Path('/workspace/work/recover-small_h2_56/review/coarse_family_protocol/synthetic');O=Path(__file__).with_name('t230_synthetic_saved_review.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def avg(v):return sum(v,F(0))/len(v)
def ident(r):return [r['kind'],r['index']]
def main():
 assert not O.exists();start=time.process_time();p=read(B/'protocol.json');g=read(B/'gate.json');f=read(B/'fixtures.json');r=read(B/'output/results.json')
 for pins in (p['pins'],g['pins']):
  for path,h in pins.items():assert sha(path)==h
 assert r['complete'] is True and r['pass_all'] is True and r['model_face_rng_calls']==0
 assert r['gate']==g and g['clearance'] is True and r['protocol_sha256']==g['protocol_sha256']==sha(B/'protocol.json') and r['runtime']==p['runtime']
 for path in (p['design_review'],g['source_review']['path']):
  rv=read(path);assert rv['complete'] is True and rv['pass_all'] is True
 assert read(p['design_review'])['execution_authorized'] is False
 assert len(r['cases'])==len(f['cases'])==8 and len(r['guards'])==len(f['guards'])==10
 row_count=0;case_stats=[]
 for c,saved in zip(f['cases'],r['cases'],strict=True):
  assert saved['name']==c['name'] and saved['passed'] is True
  A=c['A'];D=c['B'];av=[F(x['value']) for x in A];bv=[F(x['value']) for x in D];pooled=avg(av)
  keys={(x['kind'],x['label']) for x in A};means={k:avg([v for x,v in zip(A,av,strict=True) if (x['kind'],x['label'])==k]) for k in keys}
  preds=[means.get((x['kind'],x['label']),pooled) for x in D];res=[v-a for v,a in zip(bv,preds,strict=True)];base=[v-pooled for v in bv];n=len(D)
  # Pairwise identity checks centered variance independently of subtracting moments.
  centered=sum(((res[i]-res[j])**2 for i in range(n) for j in range(i)),F(0))/(n*n)
  mse=avg([v*v for v in res]);baseline=avg([v*v for v in base]);ans=saved['score']
  assert F(ans['train_mean'])==pooled and {tuple(json.loads(k)):F(v) for k,v in ans['means'].items()}==means
  for key,values in [('predictions',preds),('residuals',res)]:assert list(map(F,ans[key]))==values
  for key,value in [('residual_mean',avg(res)),('mse',mse),('baseline_mse',baseline),('centered_residual_variance',centered)]:assert F(ans[key])==value
  ratio=None if baseline==0 else mse/baseline;assert (None if ans['ratio'] is None else F(ans['ratio']))==ratio
  unseen=[ident(x) for x in D if (x['kind'],x['label']) not in keys];assert ans['unseen']==unseen
  for key,v in [('mse',mse),('baseline_mse',baseline),('ratio',ratio)]:assert (None if c['expected'][key] is None else F(c['expected'][key]))==v
  exp=c['expected']
  if 'tail' in saved:
   q=F(exp['tail_fraction'])*n;keep=-(-q.numerator//q.denominator)
   # Pairwise ranks avoid reproducing the author's sorted implementation.
   order_keys=[(-v*v,x['kind'],x['index']) for x,v in zip(D,bv,strict=True)]
   ranks=[sum(other<key for other in order_keys) for key in order_keys]
   ids=[ident(D[ranks.index(k)]) for k in range(keep)]
   assert saved['tail']==exp['tail']==ids
  if 'permuted' in saved:
   perm=exp['permutation'];assert sorted(perm)==list(range(n))
   changed=saved['permuted'];assert saved['occupancy_preserved'] is True
   for i,x in enumerate(changed):
    assert x==dict(D[i],value=str(F(D[i]['value'])),label=D[perm[i]]['label']) and D[i]['kind']==D[perm[i]]['kind']
   assert Counter((x['kind'],x['label']) for x in D)==Counter((x['kind'],x['label']) for x in changed)
  if 'smaller' in saved:
   counts=Counter(x['kind'] for x in D);small=[x for x in D if x['index']*2<counts[x['kind']]]
   assert saved['smaller']==[dict(x,value=str(F(x['value']))) for x in small]
   q=F(exp['tail_fraction'])*len(small);keep=-(-q.numerator//q.denominator)
   order_keys=[(-F(x['value'])**2,x['kind'],x['index']) for x in small];ranks=[sum(o<k for o in order_keys) for k in order_keys]
   ids=[ident(small[ranks.index(k)]) for k in range(keep)]
   intersection=[x for x in saved['tail'] if x in list(map(ident,small))]
   assert saved['smaller_tail']==exp['smaller_tail']==ids and saved['old_tail_intersection']==exp['old_tail_intersection']==intersection
  if 'positive_square_coverage' in saved:
   numerator=sum((v*v for x,v in zip(D,bv,strict=True) if x['label']=='common'),F(0));denominator=sum((v*v for v in bv),F(0))
   assert F(saved['positive_square_coverage'])==numerator/denominator==F(19,20)
  row_count+=n;case_stats.append({'name':c['name'],'validation_rows':n,'mse':str(mse),'baseline_mse':str(baseline),'centered_residual_variance':str(centered)})
 for name,row in zip(f['guards'],r['guards'],strict=True):assert row['name']==name and row['passed'] is True and row['caught']['type']=='ValueError' and row['unexpected_return_repr'] is None
 events=[json.loads(x) for x in (B/'output/progress.jsonl').read_text().splitlines()]
 expected=[{'phase':'attempt','name':'preflight'},{'phase':'result','name':'preflight'}]
 for name in [c['name'] for c in f['cases']]+f['guards']:expected += [{'phase':'attempt','name':name},{'phase':'result','name':name,'passed':True}]
 expected += [{'phase':'complete'}];assert events==expected and len(events)==39
 out={'complete':True,'pass_all':True,'task':'T230 independent exact synthetic saved audit','pins':{str(B/n):sha(B/n) for n in ('primitive.py','controls.py','fixtures.json','protocol.json','gate.json','output/results.json','output/progress.jsonl')},'checker_sha256':sha(Path(__file__)),'cases':8,'guards':10,'validation_rows':row_count,'progress_events':39,'case_checks':case_stats,'review_cpu_seconds':time.process_time()-start,'author_cpu_seconds':r['cpu_seconds'],'new_author_model_rng_calls':0,'limitations':['Finite equally weighted synthetic metrics only; no target-family regularity, random-null law or iid estimator claim.','Guard errors are checked against source and saved exception records; no author primitive replay.','Undefined zero-baseline ratios and raw-versus-centered residual distinction retained.','No real Sobol prefix, label-generation cost or numerical-application clearance.']}
 deviation=(B/'LAUNCHER_DEVIATION.md').read_text();assert 'ulimit -Hv 2097152' in deviation and 'ulimit -Ht 20' in deviation and 'Invalid argument' in deviation
 out['pins'][str(B/'LAUNCHER_DEVIATION.md')]=sha(B/'LAUNCHER_DEVIATION.md');out['pins'][str(B/'run1.log')]=sha(B/'run1.log')
 out['arithmetic_coverage_pass']=True;out['full_resource_protocol_compliance']=False
 out['resource_deviation']='Author-preserved shell record: hard limits failed before soft limits were lowered; no errexit, so Python continued. Harness verifies soft limits only; actual hard-limit values unrecorded. No rerun, no full launcher-conformance claim.'
 out['pass_all_scope']='All independent saved arithmetic, provenance, record and finite guard checks; resource protocol deviation is explicitly excluded from pass_all.'
 O.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'sha256':sha(O),'cpu':out['review_cpu_seconds']}))
if __name__=='__main__':main()
