"""Saved direct-float facet errors/counts/cancellation, no candidate evaluation."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
from collections import Counter
import hashlib,json,math,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/facet_mean_float');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();frac=lambda a:F(int(a[0]),int(a[1]))
g=read(P/'protocol.json');d=read(P/'controls.json')
for p,h in g['pins'].items():assert sha(p)==h
old=[r for r in read(g['saved'])['rows'] if r['kind']=='facet'];assert len(old)==len(d['rows'])==20 and d['complete']
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()];assert len(progress)==40
absolute=relative=0;counts=Counter();failures=[]
for i,(r,s) in enumerate(zip(d['rows'],old,strict=True)):
 assert r['id']==s['id']==g['ids'][i] and r['saved_value200']==s['value200'] and r['saved_certificate_status']==s['certificate']['status']
 assert r['inherited_exact_preparation_counts']==s['certificate']['counts'] and r['inherited_exact_preparation_bits']==s['certificate']['bits']
 assert progress[2*i]['id']==r['id'] and progress[2*i+1]['row']==r
 c=r['candidate'];v=F(c['value']);truth=F(Decimal(s['value200']['value']));e=abs(v-truth);rel=e/abs(truth) if truth else None;dg=r['diagnostics']
 assert dg['valid'] and math.isfinite(c['value']) and c['status'] in ('ok','zero_output','opposite_measure_zero')
 assert frac(dg['error'])==e and (frac(dg['relative_error']) if rel is not None else dg['relative_error'])==rel
 ab=e<=F(1,10**12);rr=(rel<=F(1,10**8)) if rel is not None else v==0
 assert ab==dg['absolute_pass'] and rr==dg['relative_pass'];absolute+=ab;relative+=rr
 terms=[F(t['contribution']) for t in c['terms']];total=0.
 for t in c['terms']:total=total+t['contribution']
 assert total.hex()==float(v).hex()
 assert (frac(dg['saved_term_cancellation_ratio']) if v else dg['saved_term_cancellation_ratio'])==(sum(map(abs,terms),F())/abs(v) if v else None)
 if s['certificate']['status']=='prepared':
  fs=s['certificate']['facets'];N=len(fs);K=sum(f['other_count']==2 for f in fs);C=2*N+2*K
  expected={'multiplications':1+N,'sqrt':1+N+K,'exact_rational_to_binary64':C,'finite_checks':C+1,'comparisons':N+K+1,'divisions':2*N+K,'additions':N}
  if K:expected.update(negations=K,atan2=K)
  assert c['counts']==expected,(r['id'],c['counts'],expected)
 else:assert c['counts']=={} and v==0 and terms==[]
 counts.update(c['counts'])
 if not rr:
  exactterms=sum(terms,F());firstpair=float(c['terms'][0]['contribution']+c['terms'][1]['contribution'])
  failures.append({'id':r['id'],'absolute_error':float(e),'relative_error':float(rel),'term_evaluation_error':float(exactterms-truth),'summation_rounding_error':float(v-exactterms),'first_pair_exact':F(firstpair)==terms[0]+terms[1],'term_cancellation_ratio':float(sum(map(abs,terms),F())/abs(v))})
assert d['valid']==20 and d['absolute_pass']==absolute==20 and d['relative_pass']==relative==19 and [r['id'] for r in failures]==['narrow40']
out={'pass':True,'valid':20,'absolute_pass':20,'relative_pass':19,'failures':failures,'named_numeric_categories':dict(counts),'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'output_sha256':sha(P/'controls.json'),'protocol_sha256':sha(P/'protocol.json'),'source_sha256':sha(P/'kernel.py'),'scope':'Exact saved decimal-reference error and threshold/counter audit; direct binary64 cancellation failure retained. No exact preparation, candidate, MP or angular replay. Counts exclude native/libm/conversion internals and exact setup. No general FP or speed claim.'}
(O/'facet_mean_float_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
