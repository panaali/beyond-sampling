"""Symbolic coefficient identity via exact Dirichlet monomial enumeration."""
from collections import Counter
from itertools import product
from pathlib import Path
import json,math,time
start=time.process_time()
def add(*args):
 d=Counter()
 for a in args:
  for k,v in a.items():d[k]+=v
 return {k:v for k,v in d.items() if v}
def scale(a,s):return {k:v*s for k,v in a.items()}
def mul(a,b):
 d=Counter()
 for i,v in a.items():
  for j,w in b.items():d[tuple(sorted(i+j))]+=v*w
 return dict(d)
var={(i,j):f'g{i}{j}' for i in range(4) for j in range(i,4)};G=[[{(var[min(i,j),max(i,j)],):1} for j in range(4)] for i in range(4)]
r=[add(*row) for row in G];s=add(*r);t=add(*(G[i][i] for i in range(4)));diag=[G[i][i] for i in range(4)]
formula=add(mul(s,s),scale(mul(s,t),2),scale(add(*(mul(v,v) for v in r)),4),mul(t,t),scale(add(*(mul(G[i][j],G[j][i]) for i,j in product(range(4),repeat=2))),2),scale(add(*(mul(diag[i],r[i]) for i in range(4))),8),scale(add(*(mul(v,v) for v in diag)),6))
exact=Counter()
for i,j,k,l in product(range(4),repeat=4):
 coeff=math.prod(math.factorial(c) for c in Counter((i,j,k,l)).values())
 for monomial,v in mul(G[i][j],G[k][l]).items():exact[monomial]+=coeff*v
assert formula==dict(exact)
mean=add(s,t);exactmean=Counter()
for i,j in product(range(4),repeat=2):
 for k,v in G[i][j].items():exactmean[k]+=(1+int(i==j))*v
assert mean==dict(exactmean)
out={'independent_four_index_monomials':256,'distinct_quadratic_coefficients':len(exact),'first_moment_denominator':20,'second_moment_denominator':840,'exact_symbolic_identity':True,'cpu_seconds':time.process_time()-start,'scope':'real-arithmetic moment identity; no cancellation/error certification for floating evaluation'}
Path(__file__).with_name('dirichlet_moments_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
