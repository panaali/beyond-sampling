"""Saved tie-control audit using exact halfline coefficients and incomplete beta CDF."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,time
import mpmath as mp
P=Path('/workspace/work/recover-small_h4_4/review/srb_tie_semantics');H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 start=time.process_time();mp.mp.dps=110;d=read(P/'results.json');g=read(P/'gate.json');p=read(P/'protocol.json');fs=read(P/'fixtures.json');events=[json.loads(s) for s in (P/'progress.jsonl').read_text().splitlines()]
 for path,h in {**p['pins'],**g['pins']}.items():assert sha(path)==h,path
 assert d['complete'] and d['gate_sha256']==sha(P/'gate.json') and d['protocol_sha256']==sha(P/'protocol.json')
 assert Counter(e['stage'] for e in events)==dict(started=1,case_started=12,case_completed=12,completed=1)
 assert events[-1]['sha256']==sha(P/'results.json')
 assert [(e['name'],e['radial']) for e in events if e['stage']=='case_started']==[(f['name'],radial) for f in fs for radial in [False,True]]
 assert [(e['name'],e['radial']) for e in events if e['stage']=='case_completed']==[(f['name'],radial) for f in fs for radial in [False,True]]
 assert d['calls']==dict(original_Q_batches=10,repaired_Q_batches=12,original_conditional_prefixes=20,repaired_conditional_prefixes=24) and d['reused_original_Gaussian_prefixes']==2
 prior=read(p['original_results'])['rows'][0]['runs'][0];errors=[];referrors=[];baselineerrors=[]
 def cdf(x,radial):
  if mp.isinf(x):return mp.mpf(x>0)
  if not radial:return mp.erfc(-x/mp.sqrt(2))/2
  if not x:return mp.mpf('.5')
  tail=mp.betainc(mp.mpf('4.5'),mp.mpf('.5'),0,1/(1+x*x),regularized=True)/2
  return tail if x<0 else 1-tail
 for i,(f,radial) in enumerate((f,r) for f in fs for r in [False,True]):
  row=d['rows'][i];assert row==read(P/f'row_{i:02d}.json') and row['name']==f['name'] and row['radial']==radial
  assert row['model']=={k:f[k] for k in ['Wi','Wh','Wo']} and row['prefixes']==f['x'] and all(not any(r) for r in f['Wh'])
  pred={};strict={}
  for sign in [-1,1]:
   logits=[sum(F(v)*max(F(0),sign*F(w)) for v,w in zip(r,f['Wi'],strict=True)) for r in f['Wo']]
   winner=max(range(10),key=lambda j:(logits[j],-j));pred[sign]=winner;strict[sign]=winner if sum(v==logits[winner] for v in logits)==1 else None
  original=row['original']
  if f['reuse_original']:
   if radial:assert original is None
   else:assert original['reused'] and original['hex']==prior['original']['hex'] and original['warnings']==prior['original_warnings']
  if f['bit_parity_expected']:assert original['hex']==row['repaired']['hex']
  for arm in [original,row['repaired']]:
   if arm is not None:assert arm['hex']==[float(v).hex() for v in arm['values']]
  for j,rawx in enumerate(f['x']):
   x=list(map(mp.mpf,rawx));order=sorted(range(9),key=lambda k:(-rawx[k],k));a,b=order[:2]
   if radial:
    norm=mp.sqrt(sum(v*v for v in x));x=[v/norm for v in x]
   m1,m2=x[a],x[b];total=mp.mpf(0);oldtotal=mp.mpf(0)
   domains=[(-mp.inf,m2,b),(m2,m1,9),(m1,mp.inf,a)]
   for lo,hi,target in domains:
    for sign,slo,shi in [(-1,-mp.inf,mp.mpf(0)),(1,mp.mpf(0),mp.inf)]:
     L,U=max(lo,slo),min(hi,shi)
     if L<U:
      mass=cdf(U,radial)-cdf(L,radial)
      if pred[sign]==target:total+=mass
      if strict[sign]==target:oldtotal+=mass
   ref=row['reference'][j];referrors.append(abs(total-mp.mpf(ref['value'])));assert referrors[-1]<mp.mpf('1e-75')
   pieces=mp.mpf(0)
   for lo,hi,label,mass in ref['pieces']:
    L,U=mp.mpf(lo),mp.mpf(hi);assert L<U
    sample=U-1 if mp.isinf(L) else L+1 if mp.isinf(U) else (L+U)/2
    sign=-1 if sample<0 else 1;target=a if sample>m1 else 9 if sample>m2 else b
    assert pred[sign]==target==label and abs(cdf(U,radial)-cdf(L,radial)-mp.mpf(mass))<mp.mpf('1e-75');pieces+=mp.mpf(mass)
   assert abs(pieces-total)<mp.mpf('1e-75')
   errors.append(abs(mp.mpf(row['repaired']['values'][j])-total));assert errors[-1]<=mp.mpf('3e-13')
   if original is not None:baselineerrors.append(abs(mp.mpf(original['values'][j])-oldtotal));assert baselineerrors[-1]<=mp.mpf('3e-13')
 out=dict(complete=True,pass_all=True,source_result_commit='9a3836d1',checker_sha256=sha(__file__),pins={n:sha(P/n) for n in ['results.json','protocol.json','gate.json','progress.jsonl']},rows=12,prefixes=24,original_new_calls=10,repaired_calls=12,reused_prefixes=2,max_repaired_error=str(max(errors)),max_saved_reference_error=str(max(referrors)),max_original_strict_semantics_error=str(max(baselineerrors)),review_cpu=time.process_time()-start,author_cpu=d['cpu_seconds'],new_Q_model_random_quadrature_calls=0,scope='Exact halfline logit identities and independent Gaussian erfc/Student incomplete-beta CDF contractions. Fixed analytic controls only; no trained correctness, outward certification, shared adoption or complete cost claim.')
 (H/'t210_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=out['review_cpu'],max_error=out['max_repaired_error'])))
if __name__=='__main__':main()
