"""Exact one-variable controls and finite-MP conditional bounds; not an estimator."""
from fractions import Fraction as F
import mpmath as mp
from scalar import absolute_mean, bound_at_rate

def dot(a,b): return sum((x*y for x,y in zip(a,b)), F(0))
def matvec(A,x): return [dot(a,x) for a in A]
def forward(Wi,Wh,Wo,inputs):
 h=[F(0)]*len(Wi)
 for x in inputs: h=[max(F(0),a+b*x) for a,b in zip(matvec(Wh,h),Wi)]
 scores=matvec(Wo,h)
 return max(range(len(scores)),key=lambda j:scores[j]),scores,h
def value(aff,x): return aff[0]+aff[1]*x
def probe(lo,hi):
 assert hi is not None and (lo is None or lo<hi)
 return hi-1 if lo is None else (lo+hi)/2
def root(aff): return -aff[0]/aff[1] if aff[1] else None
def inside(x,lo,hi): return x is not None and (lo is None or lo<x) and x<hi
def nonnegative_on(aff,lo,hi):
 # Closed endpoint checks plus the exact recession sign cover the whole interval.
 return value(aff,hi)>=0 and (aff[1]<=0 if lo is None else value(aff,lo)>=0)
def pm(x): return mp.mpf(x.numerator)/x.denominator
def cdf(x): return mp.mpf(0) if x is None else mp.erfc(-pm(x)/mp.sqrt(2))/2

def partition(Wi,Wh,Wo,prefix,b,target):
 _,_,h=forward(Wi,Wh,Wo,prefix)
 pre=list(zip(matvec(Wh,h),Wi))
 gates=sorted({r for z in pre if (r:=root(z)) is not None and r<b})
 cells=[]; allcuts=set(gates); left=None
 for right in gates+[b]:
  q=probe(left,right); mask=[value(z,q)>0 for z in pre]
  hidden=[z if g else (F(0),F(0)) for z,g in zip(pre,mask)]
  scores=[(dot(w,[z[0] for z in hidden]),dot(w,[z[1] for z in hidden])) for w in Wo]
  cuts=set()
  for i in range(len(Wo)):
   for j in range(i):
    z=tuple(scores[i][k]-scores[j][k] for k in range(2));r=root(z)
    if inside(r,left,right): cuts.add(r)
  allcuts.update(cuts); lo=left
  for hi in sorted(cuts)+[right]:
   x=probe(lo,hi); winner=max(range(len(Wo)),key=lambda j:value(scores[j],x))
   for z,g in zip(pre,mask):
    assert nonnegative_on(z if g else tuple(-v for v in z),lo,hi)
   for j in range(len(Wo)):
    diff=tuple(scores[winner][k]-scores[j][k] for k in range(2))
    assert nonnegative_on(diff,lo,hi)
    if j<winner: assert value(diff,x)>0  # ties go to the earlier index
   direct,raw,_=forward(Wi,Wh,Wo,prefix+[x])
   assert direct==winner and raw==[value(z,x) for z in scores]
   cells.append({'lo':None if lo is None else str(lo),'hi':str(hi),'probe':str(x),
    'mask':[int(g) for g in mask],'pre':[[str(v) for v in z] for z in pre],
    'scores':[[str(v) for v in z] for z in scores],'winner':winner,'correct':winner==target})
   lo=hi
  left=right
 boundaries=[]
 for x in sorted(allcuts|{b}):
  winner,scores,_=forward(Wi,Wh,Wo,prefix+[x])
  boundaries.append({'x':str(x),'winner':winner,'scores':list(map(str,scores))})
 prob=sum((cdf(F(c['hi']))-cdf(None if c['lo'] is None else F(c['lo'])) for c in cells if c['correct']),mp.mpf(0))/cdf(b)
 return {'cells':cells,'boundaries':boundaries,'probability':mp.nstr(prob,110),
  'validation_forward_calls':1+len(cells)+len(boundaries)}

def candidate(Wi,Wh,Wo,prefix,b,target):
 anchor=F(0); pred,scores,_=forward(Wi,Wh,Wo,prefix+[anchor]); rows=[]
 lower=mp.mpf(1);upper=mp.mpf(1);mean=absolute_mean(pm(b),0)
 for j in range(len(Wo)):
  if j==target:continue
  margin=scores[target]-scores[j]
  L=dot([abs(x-y) for x,y in zip(Wo[target],Wo[j])],[abs(x) for x in Wi])
  strict=j<target
  if not L:
   good=margin>0 if strict else margin>=0
   failure=mp.mpf(0 if good else 1); required=mp.mpf(1 if good else 0)
   status='constant_correct' if good else 'constant_wrong';rate=None
  else:
   m=pm(margin);ell=pm(L);budget=ell*mean
   rate=abs(m)/(ell*ell)
   # Precheck only eliminates this anchor/global-envelope Chernoff family.
   B=bound_at_rate(pm(b),[0],[ell],abs(m),rate) if abs(m)>budget else mp.mpf(1)
   if margin>0:failure=B;required=mp.mpf(1)
   elif margin<0:failure=mp.mpf(1);required=B
   else:failure=required=mp.mpf(1)
   status='nonvacuous_precheck' if abs(m)>budget else 'vacuous_precheck'
  lower-=failure;upper=min(upper,required)
  rows.append({'rival':j,'strict':strict,'margin':str(margin),'L':str(L),'status':status,
   'rate':None if rate is None else mp.nstr(rate,110),
   'failure_bound':mp.nstr(failure,110),'required_upper':mp.nstr(required,110)})
 return {'anchor':'0','anchor_prediction':pred,'rows':rows,
  'lower':mp.nstr(max(mp.mpf(0),lower),110),'upper':mp.nstr(upper,110),
  'candidate_forward_calls':1,'mean_absolute_deviation':mp.nstr(mean,110)}

def model(c,kind):
 Wi=list(map(F,(1,-1,0,0)));Wh=[[F(c if i==j else 0) for j in range(4)] for i in range(4)]
 plus=list(map(F,(1,0,0,0)));minus=list(map(F,(0,1,0,0)));zero=[F(0)]*4
 Wo=[minus,plus,zero] if kind=='base' else [plus,minus,zero] if kind=='swapped' else [plus,plus,plus]
 assert kind in ('base','swapped','tied')
 return Wi,Wh,Wo

def rational_strong_certificate():
 # For retained4,3: S=7, rival0 dominates zero rival2, L=2.
 # Phi(3)>=1/2 and the untruncated absolute-normal MGF <=2 exp(s²/2).
 # lambda=7/4 gives 4 exp(-49/8); a positive Taylor partial bounds exp below.
 z=F(49,8);term=F(1);partial=term
 for j in range(1,11):term*=z/j;partial+=term
 upper=4/partial
 assert upper<F(1,100)
 return {'terms':11,'exponent':str(z),'exp_lower':str(partial),
  'conditional_error_upper':str(upper),'conditional_accuracy_lower':str(1-upper),
  'scope':'Base c1 retained4,3 only; exact rational and real-theory certificate.'}
