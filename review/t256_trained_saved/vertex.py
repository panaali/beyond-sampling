"""Reviewer-only exact vertices; copied from independently reviewed65ac580f."""
from fractions import Fraction as F
from itertools import permutations,combinations
import time
START=time.process_time()
def cap():assert time.process_time()-START<19
def det(rows):
    n=len(rows);total=F(0)
    for perm in permutations(range(n)):
        term=F((-1)**sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n)))
        for i,j in enumerate(perm):term*=rows[i][j]
        total+=term
    return total

def independent_vertices(box,i,j,g):
    n=len(box);planes=[]
    # Use >= rows, opposite to the author's <= constraint representation.
    for k,(lo,hi) in enumerate(box):
        a=[F(0)]*n;a[k]=1;planes.append((tuple(a),lo))
        a=[F(0)]*n;a[k]=-1;planes.append((tuple(a),-hi))
    a=[F(0)]*n;a[i]=1;a[j]=-1;planes.append((tuple(a),g))
    for k in range(n):
        if k in(i,j):continue
        a=[F(0)]*n;a[j]=1;a[k]=-1;planes.append((tuple(a),g))
    vertices=set();nonzero=0;systems=0
    for ids in combinations(range(len(planes)),n):
        cap();systems+=1;A=[planes[k][0] for k in ids];b=[planes[k][1] for k in ids];d=det(A)
        if not d:continue
        nonzero+=1
        x=tuple(det([[b[r] if c==col else A[r][c] for c in range(n)] for r in range(n)])/d for col in range(n))
        if all(sum(a*v for a,v in zip(normal,x))>=rhs for normal,rhs in planes):vertices.add(x)
    return planes,sorted(vertices),systems,nonzero

