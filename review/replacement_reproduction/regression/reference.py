"""Exact Fraction Gram equations and Cramer solves, no scientific library."""
from fractions import Fraction as F
from itertools import permutations

def determinant(a,counts):
    counts['determinants']+=1;n=len(a);out=F()
    for p in permutations(range(n)):
        v=F(-1 if sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))%2 else 1)
        for i,j in enumerate(p):v*=a[i][j]
        out+=v
    return out

def solve(a,b,counts):
    counts['systems']+=1;den=determinant(a,counts)
    if den==0:raise ValueError('singular exact system')
    n=len(a);q=len(b[0]);out=[[F() for _ in range(q)] for _ in range(n)]
    for j in range(q):
        for k in range(n):
            z=[row[:] for row in a]
            for i in range(n):z[i][k]=b[i][j]
            out[k][j]=determinant(z,counts)/den
    return out

def fit(case,counts):
    ids=sorted(set(case['ids']));v=case['v'];y=case['y'];d=len(v[0]);q=len(y[0]);grams=[];cross=[];ns=[]
    for label in ids:
        rows=[i for i,c in enumerate(case['ids']) if c==label];ns.append(len(rows))
        grams.append([[sum((F(v[i][j])*v[i][k] for i in rows),F()) for k in range(d)] for j in range(d)])
        cross.append([[sum((F(v[i][j])*y[i][k] for i in rows),F()) for k in range(q)] for j in range(d)])
    G=[[sum((g[i][j] for g in grams),F()) for j in range(d)] for i in range(d)];B=[[sum((b[i][j] for b in cross),F()) for j in range(q)] for i in range(d)]
    def matrix(g,pooled):
        trace=sum(g[i][i] for i in range(d));ridge=trace/F(d*10**9) if pooled else (trace/d+F(1,10**12))/10**6
        return [[g[i][j]+(ridge if i==j else 0) for j in range(d)] for i in range(d)]
    systems=[{'label':'pooled','matrix':matrix(G,True),'rhs':B}]+[{'label':'cell:'+str(label),'matrix':matrix(g,False),'rhs':b} for label,g,b in zip(ids,grams,cross)]
    coefs=[solve(s['matrix'],s['rhs'],counts) for s in systems]
    return {'ids':ids,'counts':ns,'grams':grams,'cross':cross,'systems':systems,'pooled':coefs[0],'coef':coefs[1:]}

def predict(model,c,v):
    out=[];seen=[]
    for label,row in zip(c,v):
        yes=label in model['ids'];w=model['coef'][model['ids'].index(label)] if yes else model['pooled'];seen.append(yes)
        out.append([sum((F(x)*z for x,z in zip(row,col)),F()) for col in zip(*w)])
    return out,seen
