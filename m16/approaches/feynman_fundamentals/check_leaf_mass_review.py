"""Independent exact saved-ray certificates and analytic chamber totals."""
from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
import collections,json,math,time
import numpy as np
import mpmath as mp
start=time.process_time();here=Path(__file__).parent
src=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review')
d=json.loads((src/'mass_results.json').read_text());geometry=json.loads((src/'capture.json').read_text());arrays=np.load(src/'captured.npz');methods=collections.Counter();output=[]
perms=[(p,(-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))) for p in permutations(range(4))]
def determinant(A):return sum(s*math.prod(A[i][p[i]] for i in range(4)) for p,s in perms)
for row in d['pairs']:
 p0,p1=[geometry['parents'][i] for i in row['parents']];assert p0['perm']==p1['perm'] and p0['sigma']==p1['sigma']
 rank=p0['perm'].index(0)+1;prob=F(sum(math.comb(4,k) for k in range(rank,5)),16);truth=(prob if p0['sigma']>0 else 1-prob)/24
 assert F(row['analytic_mass'])==truth
 ids=set(np.flatnonzero(np.isin(arrays['parent_ids'],row['parents'])).tolist());assert ids=={r['index'] for r in row['records']}
 for r in row['records']:
  A=[[F(float(x)) for x in v] for v in arrays['rays'][r['index']]];G=[[sum(x*y for x,y in zip(a,b)) for b in A] for a in A];c=r['radial_certificate'];a,b,det=map(F,[c['a'],c['b'],c['det']]);assert det==determinant(A) and det!=0;assert b==max(G[i][i] for i in range(4));assert 0<a<=b
  methods[c['method']]+=1
  if c['method']=='exact_acute':assert min(x for v in G for x in v)>=0 and a==min(G[i][i] for i in range(4))/4
  else:
   y=list(map(F,c['y']));slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)];assert min(y)>=0 and sum(y)==1 and min(slack)>=0 and all(y[i]*slack[i]==0 for i in range(4));assert slack==list(map(F,c['slack']))
 assert math.fsum(r['source_value'] for r in row['records'])==row['source_sum']
 assert sum(r['source_value']<0 for r in row['records'])==row['negative_source_values']
 for rule in row['rules']:
  assert len(rule['values'])==len(ids) and min(rule['values'])>=0;assert math.fsum(rule['values'])==rule['sum'];assert math.fsum(rule['bounds'])==rule['conditional_integration_bound']
  assert abs(rule['sum']-float(truth))<=rule['conditional_integration_bound']
  with mp.workdps(60):
   for r,claimed in zip(row['records'],rule['bounds']):
    c=r['radial_certificate'];a,b,det=[mp.mpf(F(c[k]).numerator)/F(c[k]).denominator for k in ('a','b','det')];alpha=(a+b)/2;delta=mp.sqrt(a*b);rho=(mp.sqrt(b)-mp.sqrt(a))/(mp.sqrt(b)+mp.sqrt(a));n=rule['order'];v=abs(det)/(3*mp.pi**2)*rho**(n-1)/delta**3*((alpha+(n-1)*delta)/(1-rho)+delta*rho/(1-rho)**2)
    assert abs(float(v)-claimed)<=1e-14*max(float(v),1e-300)
 output.append(dict(parents=row['parents'],simplices=len(ids),truth=str(truth),source_error=row['source_sum']-float(truth),negative_source_values=row['negative_source_values'],positive32_error=row['rules'][-1]['sum']-float(truth),bound=row['rules'][-1]['conditional_integration_bound']))
out=dict(pairs=output,exact_radial_certificates=dict(methods),review_cpu=time.process_time()-start,scope='Exact dyadic captured-ray bounds and saved numerical sums. Integration-only bounds; no floating geometry/full-model replay.')
(here/'leaf_mass_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
