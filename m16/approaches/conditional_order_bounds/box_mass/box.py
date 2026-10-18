"""Whole top-two boxes: exact sufficient vacuity and rational Gaussian mass."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'prefix_wedge_remainder'))
from rational_bounds import I,calc,pi_interval,sqrt_interval,accounting,serial,COUNTS

SERIES_UPPER_DEGREE=32

@lru_cache(maxsize=None)
def gaussian_cdf(x):
 x=F(x);assert -2<=x<=2
 COUNTS['gaussian_cdf_unique_calls']+=1
 if x<0:return 1-gaussian_cdf(-x)
 if x==0:return I.point(F(1,2))
 # Taylor's remainder sign holds for every z>=0, not just decreasing terms.
 z=calc('div',calc('sub',F(0),calc('mul',x,x)),F(2))
 term=total=x;lower=None
 for k in range(1,SERIES_UPPER_DEGREE+1):
  term=calc('div',calc('mul',calc('mul',term,z),F(2*k-1)),F(k*(2*k+1)))
  total=calc('add',total,term)
  if k==31:lower=total
 assert lower is not None and lower<=total
 COUNTS['integrated_gaussian_series_terms']+=33
 norm=1/sqrt_interval(2*pi_interval())
 return F(1,2)+norm*I(lower,total)

def per_label_mass():
 p0,p1,p2=(gaussian_cdf(F(j)) for j in range(3))
 delta=p2-p1;rank=p1*p1*p1-p0*p0*p0
 assert delta.lo>0 and rank.lo>0
 return delta*rank/3

def sufficient(L,largest,second):
 assert len(L)==4 and largest!=second and largest in range(4) and second in range(4)
 assert all(x>=0 for x in L)
 omitted=[k for k in range(4) if k not in (largest,second)]
 S=calc('add',L[omitted[0]],L[omitted[1]])
 A=calc('add',calc('mul',F(2),L[largest]),L[second])
 threshold=calc('div',S,F(4))
 positive=S>0; relation=A<=threshold
 COUNTS['sufficient_pair_comparisons']+=2
 return {'largest':largest,'second':second,'omitted':omitted,'L_by_input':L,
  'anchor_margin_abs_upper':A,'allowance_mean_strict_lower':threshold,
  'positive_omitted_allowance':positive,'certified_vacuous':positive and relation,
  'retained_to_threshold_ratio':None if not S else calc('div',A,threshold)}

def decode_model_bounds(model):
 assert model['complete'] and model['model']['T']==4
 rows=model['envelope']['rows'];assert [r['age'] for r in rows]==list(range(4))
 result={}
 for row in rows:
  pairs=row['pairs'];assert len(pairs)==6
  assert {(p['a'],p['b']) for p in pairs}=={(a,b) for a in range(4) for b in range(a)}
  for p in pairs:
   key=(p['a'],p['b']);value=F(*p['bound']);assert value>=0
   result.setdefault(key,[None]*4)[3-row['age']]=value
 assert all(all(x is not None for x in L) for L in result.values())
 return result

def se_threshold(mass):
 # This compares certificate width with maximal sampling standard error, not MSE.
 if mass.lo==0:return None
 square=calc('mul',mass.lo,mass.lo)
 cutoff=calc('div',F(1),calc('mul',F(4),square))
 n=cutoff.numerator//cutoff.denominator+1
 assert 4*n*mass.lo*mass.lo>1
 return {'samples_at_or_above':n,'condition':'1/(2 sqrt(N)) < certified undecided mass lower endpoint',
  'scope':'Uniform Bernoulli standard-error comparison only; no matched-compute or actual-error lower bound.'}
