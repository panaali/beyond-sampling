"""Exact radial certificates and positive Gaussian simplex quadrature."""
from fractions import Fraction as F
import math
import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss

def solve(A,b):
    n=len(b);M=[list(row)+[rhs] for row,rhs in zip(A,b)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if M[i][j]);M[j],M[pivot]=M[pivot],M[j];v=M[j][j];M[j]=[x/v for x in M[j]]
        for i in range(n):
            if i!=j:
                v=M[i][j];M[i]=[x-v*y for x,y in zip(M[i],M[j])]
    return [row[-1] for row in M]

def determinant(A):
    M=[list(row) for row in A];d=F(1)
    for j in range(len(M)):
        pivot=next((i for i in range(j,len(M)) if M[i][j]),None)
        if pivot is None:return F(0)
        if pivot!=j:M[j],M[pivot]=M[pivot],M[j];d=-d
        v=M[j][j];d*=v
        for i in range(j+1,len(M)):
            c=M[i][j]/v
            for k in range(j+1,len(M)):M[i][k]-=c*M[j][k]
    return d

def bounds(R):
    A=[[F(float(x)) for x in row] for row in R];G=[[sum(x*y for x,y in zip(a,b)) for b in A] for a in A];det=determinant(A);assert det
    candidates=[]
    for mask in range(1,16):
        S=[i for i in range(4) if mask>>i&1];z=solve([[G[i][j] for j in S] for i in S],[F(1)]*len(S));den=sum(z);assert den>0;y=[x/den for x in z]
        if min(y)>=0:candidates.append((1/den,S,y))
    a,S,w=min(candidates,key=lambda x:x[0]);y=[F(0)]*4
    for i,v in zip(S,w):y[i]=v
    slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)];assert min(slack)>=0 and all(y[i]*slack[i]==0 for i in range(4));b=max(G[i][i] for i in range(4));assert 0<a<=b
    return dict(a=str(a),b=str(b),det=str(det),y=list(map(str,y)),slack=list(map(str,slack)),feasible_faces=len(candidates))

def probability_bound(cert,n):
    def m(s):v=F(s);return mp.mpf(v.numerator)/v.denominator
    with mp.workdps(50):
        a,b=m(cert['a']),m(cert['b']);alpha=(a+b)/2;delta=mp.sqrt(a*b);rho=(mp.sqrt(b)-mp.sqrt(a))/(mp.sqrt(b)+mp.sqrt(a));assert 0<=rho<1
        N=n-2;tail=2*rho**(N+1)/delta**3*(alpha/(1-rho)+delta*((N+1)/(1-rho)+rho/(1-rho)**2));v=abs(m(cert['det']))*tail/(6*mp.pi**2)
        return dict(bound=float(v),decimal=mp.nstr(v,50),rho=mp.nstr(rho,50))

def integrate(R,n):
    x,w=leggauss(n);x=(x+1)/2;w=w/2;u,v,z=np.meshgrid(x,x,x,indexing='ij');a,b,c=np.meshgrid(w,w,w,indexing='ij');y=np.stack([u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)],axis=-1).reshape(-1,4);weights=(a*b*c*(1-u)**2*(1-v)).ravel();p=y@R
    return float(abs(np.linalg.det(R))*np.dot(weights,1/np.sum(p*p,axis=1)**2)/(2*math.pi**2))
