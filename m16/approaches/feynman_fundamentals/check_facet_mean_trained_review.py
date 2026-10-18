"""Saved544 heuristic decisions/errors and independent sequential pooled means."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
from collections import Counter
import json,hashlib,math,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/facet_mean_float');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();frac=lambda a:F(int(a[0]),int(a[1]));dec=lambda x:F(Decimal(x))
outpath=O/'facet_mean_trained_review_results.json';assert not outpath.exists()
g=read(P/'trained_protocol.json');d=read(P/'trained.json');s=read(P/'summary.json');old=read(g['saved'])
for p,h in g['pins'].items():assert sha(p)==h,p
assert d['complete'] and old['complete'] and len(d['rows'])==len(old['rows'])==544 and len(d['means'])==16
progress=[json.loads(x) for x in (P/'trained_progress.jsonl').read_text().splitlines()];assert len(progress)==1105 and [r['row'] for r in progress if r['stage']=='result']==d['rows'] and [r['row'] for r in progress if r['stage']=='mean_result']==d['means']
def diagnostic(value,truth,diag,threshold):
 e=abs(F(value)-truth);rel=e/abs(truth) if truth else None
 assert diag['valid']==(value>=0) and frac(diag['error'])==e and diag['absolute_pass']==(value>=0 and e<=threshold)
 assert (frac(diag['relative_error']) if rel is not None else diag['relative_error'])==rel and diag['relative_pass']==(value>=0 and (rel<=F(1,10**8) if rel is not None else value==0.))
 return e,rel
fallbacks=[];decode=0;counts=Counter();prep=Counter();maxerr=F(0);maxrel=F(0);basefail=[];truths=[]
for k,(r,t) in enumerate(zip(d['rows'],old['rows'],strict=True)):
 assert (r['sector'],r['unit'])==(k//16,k%16)==(t['sector'],t['unit']);a=r['candidate'];b=a['base'];v=b['value'];terms=b['terms'];truth=dec(t['value200']['value']);truths.append(truth);counts.update(b['counts']);prep.update(t['facet']['counts'])
 if b['status'] in ('zero_output','opposite_measure_zero'):expected={'fallback':False,'reason':'exact_zero_status','sumabs':0.}
 elif b['status']!='ok' or v is None or not math.isfinite(v) or v<=0:expected={'fallback':True,'reason':'failed_or_nonpositive_prepared','sumabs':None}
 else:
  total=0.
  for z in terms:total=total+abs(z['contribution'])
  indicator=(64.*2.**-53)*total;threshold=1e-10*abs(v);expected={'fallback':indicator>threshold,'reason':'signed_conditioning','sumabs':total,'indicator':indicator,'threshold':threshold,'heuristic':True}
 assert a['decision']==expected
 if expected['fallback']:
  assert a['mp_evaluations']==1 and a['status']=='mp80_ok';fallbacks.append([r['sector'],r['unit']]);n=sum(2+2*(f['other_count']==2) for f in t['facet']['facets']);assert a['rational_decodes']==n;decode+=n
  assert float(a['mp80']['value']).hex()==float(a['value']).hex() and abs(dec(a['mp80']['value'])-truth)<F(1,10**70)
  assert len(a['mp80']['terms'])==len(t['value200']['terms'])
  for u,w in zip(a['mp80']['terms'],t['value200']['terms'],strict=True):
   for key in ('probability','contribution'):assert abs(dec(u[key])-dec(w[key]))<F(1,10**70)
 else:assert a['mp_evaluations']==a['rational_decodes']==0 and a['value']==v and a['status']==b['status']
 e,rel=diagnostic(a['value'],truth,r['diagnostics'],F(1,10**12));maxerr=max(maxerr,e);maxrel=max(maxrel,rel or F(0));diagnostic(v,truth,r['baseline_diagnostics'],F(1,10**12))
 if not r['baseline_diagnostics']['relative_pass']:basefail.append([r['sector'],r['unit']])
 assert frac(r['diagnostics']['finite_base_term_sumabs'])==sum((abs(F(z['contribution'])) for z in terms),F(0))
meanrows=[]
for u,m in enumerate(d['means']):
 truth=sum((truths[k*16+u] for k in range(34)),F(0));total=direct=0.
 for k in range(34):total+=d['rows'][k*16+u]['candidate']['value'];direct+=d['rows'][k*16+u]['candidate']['base']['value']
 assert m['unit']==u and total.hex()==m['value'].hex() and frac(m['reference_exact_saved_sum'])==truth and m['published_mean']==old['summary']['mean'][u]
 assert frac(m['reference_sum_discrepancy'])==abs(truth-dec(m['published_mean']))<F(1,10**180)
 e,rel=diagnostic(total,truth,m['diagnostics'],F(1,10**10));sr=s['retrospective_direct_mean_rows'][u];de=abs(F(direct)-truth);dr=de/abs(truth)
 assert sr['direct_mean_hex']==direct.hex() and frac(sr['error'])==de and frac(sr['relative_error'])==dr and sr['absolute_pass']==(de<=F(1,10**10)) and sr['relative_pass']==(dr<=F(1,10**8)) and sr['hybrid_error']==m['diagnostics']['error']
 meanrows.append({'unit':u,'hybrid_error':float(e),'direct_error':float(de),'direct_relative_error':float(dr)})
assert len(fallbacks)==d['mp80_evaluations']==s['fallbacks']==155 and decode==d['rational_decodes']==1860 and len(basefail)==31
assert dict(counts)==d['direct_named_partial_counts']==s['partial_float_counts'] and dict(prep)==old['counts']['facet_exact_named'] and d['inherited_preparation_counts']==old['counts']
assert d['cell_valid']==d['cell_absolute_pass']==d['cell_relative_pass']==544 and d['baseline_absolute_pass']==544 and d['baseline_relative_pass']==513 and d['mean_absolute_pass']==d['mean_relative_pass']==s['direct_pooled_absolute_pass']==s['direct_pooled_relative_pass']==16
assert s['hybrid_max_cell_absolute_error']==float(maxerr) and s['hybrid_max_cell_relative_error']==float(maxrel) and s['direct_max_pooled_absolute_error']==max(r['direct_error'] for r in meanrows) and s['hybrid_max_pooled_absolute_error']==max(r['hybrid_error'] for r in meanrows)
assert d['candidate_timers_sum']==sum(r['candidate_cpu_seconds'] for r in d['rows']) and d['cpu_seconds']>=d['candidate_timers_sum']
out={'pass':True,'cells':544,'fallbacks':155,'rational_decodes':decode,'direct_relative_failures':basefail,'all_direct_and_hybrid_pooled_means_pass':True,'rows':meanrows,'max_hybrid_cell_error':float(maxerr),'max_hybrid_cell_relative_error':float(maxrel),'named_partial_counts':dict(counts),'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'result_sha256':sha(P/'trained.json'),'summary_sha256':sha(P/'summary.json'),'protocol_sha256':sha(P/'trained_protocol.json'),'scope':'Saved heuristic/cell/MP-term and pooled arithmetic audit only. No candidate, preparation or reference rerun.155 actual fallbacks repair31 cell-relative failures but direct means already pass fixed pooled tolerances; no necessity, generic FP or speed/fullcost claim.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
