from pathlib import Path
from fractions import Fraction as F
import hashlib,json,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
import mpmath as mp
mp.mp.dps=160
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/conditional_order_bounds');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((P/'network_protocol.json').read_text());r=json.loads((P/'network_results.json').read_text());assert r['complete'] and r['protocol_sha256']==sha(P/'network_protocol.json')
for file,h in p['pins'].items():assert sha(Path(file))==h,file
progress=[json.loads(x) for x in (P/'network_progress.jsonl').read_text().splitlines()];assert len(progress)==14 and [x['row'] for x in progress if x['stage']=='result']==r['rows']
mm=lambda x:mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(x)
Phi=lambda x:(1+mp.erf(mm(x)/mp.sqrt(2)))/2
phi=lambda x:mp.exp(-mm(x)**2/2)/mp.sqrt(2*mp.pi)
counts=[0,0];outrows=[]
for row,case in zip(r['rows'],p['cases'],strict=True):
 assert row['case']==case and row['passed'];name,c,kind,xx=case;x0,x1=map(F,xx);b=min(x0,x1);target=0 if x0<x1 else 1;S=c*c*x0+c*x1;threshold=-S/2
 assert row['target']==target and F(row['truncation'])==b and F(row['analytic_threshold'])==threshold
 wi=list(map(F,row['weights'][0]));wh=[[F(x) for x in z] for z in row['weights'][1]];wo=[[F(x) for x in z] for z in row['weights'][2]]
 assert wi==[1,-1,0,0] and wh==[[c if i==j else 0 for j in range(4)] for i in range(4)]
 plus=[F(1),F(0),F(0),F(0)];minus=[F(0),F(1),F(0),F(0)];zero=[F(0)]*4
 assert wo==([minus,plus,zero] if kind=='base' else [plus,minus,zero] if kind=='swapped' else [plus]*3)
 v=row['validation'];cells=v['cells'];bounds=v['boundaries'];expected_roots=sorted({z for z in (-S,F(0),threshold) if z<b}) if kind!='tied' else sorted({z for z in (-S,F(0)) if z<b})
 # Check all actual cells exactly; tied cases can retain redundant pair roots only if slope differs (here never).
 assert [F(z['hi']) for z in cells]==expected_roots+[b]
 total=mp.mpf(0);last=None
 for cell in cells:
  lo=None if cell['lo'] is None else F(cell['lo']);hi=F(cell['hi']);x=F(cell['probe']);assert lo==last and (lo is None or lo<x) and x<hi;last=hi
  pre=[(S,F(1)),(F(0),F(-1)),(F(0),F(0)),(F(0),F(0))]
  assert [[F(z) for z in a] for a in cell['pre']]==[list(a) for a in pre]
  active=[a+d*x>0 for a,d in pre];assert cell['mask']==list(map(int,active))
  forms=[pre[i] if active[i] else (F(0),F(0)) for i in range(4)]
  scores=[(sum(w[i]*forms[i][0] for i in range(4)),sum(w[i]*forms[i][1] for i in range(4))) for w in wo]
  assert [[F(z) for z in a] for a in cell['scores']]==[list(a) for a in scores]
  vals=[a+d*x for a,d in scores];winner=min(i for i,z in enumerate(vals) if z==max(vals));assert cell['winner']==winner and cell['correct']==(winner==target)
  for j,(a,d) in enumerate(scores):
   aa=scores[winner][0]-a;dd=scores[winner][1]-d;assert aa+dd*hi>=0 and (dd<=0 if lo is None else aa+dd*lo>=0)
  if cell['correct']:total+=(Phi(hi)-(0 if lo is None else Phi(lo)))/Phi(b)
 assert [F(e['x']) for e in bounds]==expected_roots+[b]
 for edge in bounds:
  x=F(edge['x']);h=[max(S+x,0),max(-x,0),F(0),F(0)];scores=[sum(wi*hi for wi,hi in zip(w,h)) for w in wo]
  winner=min(i for i,z in enumerate(scores) if z==max(scores));assert list(map(F,edge['scores']))==scores and edge['winner']==winner
 analytic=(Phi(b)-Phi(threshold))/Phi(b)
 if kind=='swapped':analytic=1-analytic
 if kind=='tied':analytic=mp.mpf(target==0)
 assert abs(total-analytic)<mp.mpf('1e-130') and abs(total-mp.mpf(v['probability']))<mp.mpf('1e-100') and abs(analytic-mp.mpf(row['analytic_probability']))<mp.mpf('1e-100')
 cand=row['candidate'];scores=[sum(wi*hi for wi,hi in zip(w,[S,F(0),F(0),F(0)])) for w in wo];pred=min(i for i,z in enumerate(scores) if z==max(scores));assert cand['anchor_prediction']==pred and cand['anchor']=='0'
 mean=(2*phi(0)-phi(b))/Phi(b);assert abs(mean-mp.mpf(cand['mean_absolute_deviation']))<mp.mpf('1e-100');low=mp.mpf(1);high=mp.mpf(1)
 assert [e['rival'] for e in cand['rows']]==[j for j in range(3) if j!=target]
 for e in cand['rows']:
  j=e['rival'];margin=scores[target]-scores[j];L=sum(abs(a-z)*abs(w) for a,z,w in zip(wo[target],wo[j],wi));assert F(e['margin'])==margin and F(e['L'])==L and e['strict']==(j<target)
  if L==0:
   good=margin>0 if j<target else margin>=0;failure=mp.mpf(not good);required=mp.mpf(good);assert e['rate'] is None and e['status']==('constant_correct' if good else 'constant_wrong')
  else:
   rate=abs(mm(margin))/mm(L)**2;assert abs(rate-mp.mpf(e['rate']))<mp.mpf('1e-100')
   improves=abs(mm(margin))>mm(L)*mean;assert e['status']==('nonvacuous_precheck' if improves else 'vacuous_precheck')
   if improves:
    s=rate*mm(L);mgf=mp.exp(s*s/2)*(Phi(s)+Phi(mm(b)-s)-Phi(-s))/Phi(b);B=min(mp.mpf(1),mp.exp(-rate*abs(mm(margin)))*mgf)
   else:B=mp.mpf(1)
   failure=B if margin>0 else mp.mpf(1);required=B if margin<0 else mp.mpf(1)
  assert abs(failure-mp.mpf(e['failure_bound']))<mp.mpf('1e-100') and abs(required-mp.mpf(e['required_upper']))<mp.mpf('1e-100');low-=failure;high=min(high,required)
 low=max(mp.mpf(0),low);assert abs(low-mp.mpf(cand['lower']))<mp.mpf('1e-100') and abs(high-mp.mpf(cand['upper']))<mp.mpf('1e-100')
 assert low<=total+mp.mpf('1e-100') and total<=high+mp.mpf('1e-100');assert v['validation_forward_calls']==1+len(cells)+len(bounds) and cand['candidate_forward_calls']==1
 counts[0]+=len(cells);counts[1]+=len(bounds);outrows.append(dict(name=name,lower=float(low),upper=float(high),analytic_probability=float(analytic)))
assert counts==[25,25] and r['candidate_forward_calls']==7 and r['validation_forward_calls']==57 and r['validation_cells']==25 and r['validation_boundaries']==25
cert=r['rational_strong_certificate'];z=F(49,8);partial=sum(z**j/F(mathfactor) for j,mathfactor in enumerate([1,1,2,6,24,120,720,5040,40320,362880,3628800]));upper=4/partial
assert cert['terms']==11 and F(cert['exponent'])==z and F(cert['exp_lower'])==partial and F(cert['conditional_error_upper'])==upper<F(1,100) and F(cert['conditional_accuracy_lower'])==1-upper
result=dict(complete=True,cases=7,cells=25,boundaries=25,candidate_forwards_retained=7,validation_forwards_retained=57,rows=outrows,rational_error_upper=str(upper),author_sha256=sha(P/'network_results.json'),review_cpu=time.process_time()-start,scope='Independent exact saved affine cells/boundaries/model algebra/first-index outputs,160-digit alternate CDF/MGF contractions and rational Taylor certificate. No author neural forwards or partition replay. Finite MP bounds diagnostic; rational certificate applies only fixed dominated-rival synthetic case.')
(O/'t189_network_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
