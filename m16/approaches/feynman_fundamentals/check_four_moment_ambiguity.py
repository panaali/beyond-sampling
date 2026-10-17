"""Exact Fraction falsifier for universal four-cumulant argmax identification."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import json

nodes=(-3,-2,-1,1,2,3)
weights=([F(n,30) for n in (4,9,0,10,1,6)],
         [F(n,30) for n in (6,1,10,0,9,4)])
moments=[[sum(w*x**k for w,x in zip(p,nodes)) for k in range(5)] for p in weights]
assert moments[0]==moments[1]
assert all(w>=0 for p in weights for w in p)
def uniform(k):return F(0) if k%2 else F(1,10)**k/F(k+1)
def first(a,mu):return sum(F(comb(a,k))*mu[k]*uniform(a-k) for k in range(a+1))
def third(c):return sum(F(comb(c,k))*(-10)**(c-k)*uniform(k) for k in range(c+1))
mixed=[]
for a in range(5):
    for b in range(5-a):
        for c in range(5-a-b):
            values=[first(a,mu)*uniform(b)*third(c) for mu in moments]
            assert values[0]==values[1]
            mixed.append(dict(powers=[a,b,c],value=str(values[0])))
winners=[[sum(w for x,w in zip(nodes,p) if x>0),sum(w for x,w in zip(nodes,p) if x<0),F(0)] for p in weights]
assert winners==[[F(17,30),F(13,30),F(0)],[F(13,30),F(17,30),F(0)]]
gap=abs(winners[0][0]-winners[1][0]);assert gap==F(2,15)
out=dict(nodes=nodes,weights=[[str(w) for w in p] for p in weights],
         common_scalar_moments=[str(x) for x in moments[0]],
         common_scalar_cumulants=[str(moments[0][1]),str(moments[0][2]),str(moments[0][3]),str(moments[0][4]-3*moments[0][2]**2)],
         jitter_half_width='1/10',common_covariance_diagonal=[str(F(14,3)+F(1,300)),'1/300','1/300'],
         equal_mixed_moments=mixed,winner_vectors=[[str(x) for x in p] for p in winners],
         max_coordinate_gap=str(gap),universal_minimax_error_lower_bound=str(gap/2),
         scope='Arbitrary full-dimensional atom-free logit laws; not a construction in the specified Gaussian-input ReLU RNN family.')
Path(__file__).with_name('four_moment_ambiguity_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='equal_mixed_moments'},indent=2))
