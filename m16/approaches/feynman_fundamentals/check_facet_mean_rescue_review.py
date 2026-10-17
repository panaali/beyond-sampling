"""Saved heuristic decisions, exact errors and actual fallback inventory."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,math,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/facet_mean_float');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();frac=lambda a:F(int(a[0]),int(a[1]))
g=read(P/'rescue_protocol.json');d=read(P/'rescue_controls.json');base=read(P/'controls.json')['rows'];old=[r for r in read(g['saved'])['rows'] if r['kind']=='facet']
for p,h in g['pins'].items():assert sha(p)==h
assert d['complete'] and len(d['rows'])==len(base)==len(old)==20
progress=[json.loads(x) for x in (P/'rescue_progress.jsonl').read_text().splitlines()];assert len(progress)==40
fallbacks=[];decodes=0
for i,(r,b,s) in enumerate(zip(d['rows'],base,old,strict=True)):
 assert r['id']==b['id']==s['id'] and r['baseline_diagnostics']==b['diagnostics'] and r['candidate']['base']==b['candidate'] and progress[2*i+1]['row']==r
 a=r['candidate'];v=b['candidate']['value'];decision=a['decision'];terms=b['candidate']['terms']
 if b['candidate']['status'] in ('zero_output','opposite_measure_zero'):expected={'fallback':False,'reason':'exact_zero_status','sumabs':0.}
 elif b['candidate']['status']!='ok' or v is None or not math.isfinite(v) or v<=0:expected={'fallback':True,'reason':'failed_or_nonpositive_prepared','sumabs':None}
 else:
  total=0.
  for t in terms:total=total+abs(t['contribution'])
  indicator=(64.*2.**-53)*total;threshold=1e-10*abs(v);expected={'fallback':indicator>threshold,'reason':'signed_conditioning','sumabs':total,'indicator':indicator,'threshold':threshold,'heuristic':True}
 assert expected==decision
 if decision['fallback']:
  assert a['mp_evaluations']==1 and a['status']=='mp80_ok';fallbacks.append(r['id']);count=sum(2+2*(f['other_count']==2) for f in s['certificate']['facets']);assert count==a['rational_decodes'];decodes+=count
  assert float(a['mp80']['value']).hex()==float(a['value']).hex() and len(a['mp80']['terms'])==len(s['value200']['terms'])
  for u,w in zip(a['mp80']['terms'],s['value200']['terms'],strict=True):
   for key in ('probability','contribution'):assert abs(F(Decimal(u[key]))-F(Decimal(w[key])))<F(1,10**75)
  assert abs(F(Decimal(a['mp80']['value']))-F(Decimal(s['value200']['value'])))<F(1,10**75)
 else:assert a['mp_evaluations']==a['rational_decodes']==0 and a['value']==b['candidate']['value'] and a['status']==b['candidate']['status']
 truth=F(Decimal(s['value200']['value']));error=abs(F(a['value'])-truth);rel=error/abs(truth) if truth else None;diag=r['diagnostics']
 assert diag['valid'] and frac(diag['error'])==error and error<=F(1,10**12) and diag['absolute_pass']
 assert (frac(diag['relative_error']) if rel is not None else diag['relative_error'])==rel and diag['relative_pass'] and ((rel<=F(1,10**8)) if rel is not None else a['value']==0.)
assert fallbacks==['quadrant_negative_sum','narrow40'] and decodes==d['rational_decodes']==24 and d['mp80_evaluations']==2 and d['reused_float_rows']==20 and d['valid']==d['absolute_pass']==d['relative_pass']==20
out={'pass':True,'cases':20,'absolute_pass':20,'relative_pass':20,'fallbacks':fallbacks,'actual_mp80_evaluations':2,'rational_decodes':24,'reused_direct_rows':20,'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'output_sha256':sha(P/'rescue_controls.json'),'protocol_sha256':sha(P/'rescue_protocol.json'),'source_sha256':sha(P/'hybrid.py'),'scope':'Saved heuristic trigger/error and actual fallback-call/term inventory only. No MP/candidate/preparation replay. Original direct narrow40 failure retained; heuristic is not a rigorous error certificate, full native cost and generic correctness remain open.'}
(O/'facet_mean_rescue_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
