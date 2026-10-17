"""Independent T3 truth: ordered top-two charts, cyclic polygons, exact-zero ownership."""
import math
import numpy as np
E=np.eye(3)

def cut(poly,n):
    if not poly:return []
    d=[float(n@v) for v in poly]
    if min(d)>=0:return poly
    if max(d)<=0:return []
    out=[]
    for i,p in enumerate(poly):
        j=(i+1)%len(poly);a,b=d[i],d[j]
        if a>=0:out.append(p)
        if (a<0<b) or (b<0<a):
            t=a/(a-b);v=(1-t)*p+t*poly[j];out.append(v/np.linalg.norm(v))
    return out if len(out)>=3 else []

def area(poly):
    if len(poly)<3:return 0.
    a=poly[0];v=0.
    for i in range(1,len(poly)-1):
        b,c=poly[i],poly[i+1]
        numerator=abs(float(np.linalg.det(np.stack([a,b,c]))))
        denominator=1.+float(a@b)+float(a@c)+float(b@c)
        v+=2*math.atan2(numerator,denominator)
    return v/(4*math.pi)

def split(poly,rows):
    parts=[(poly,[])]
    for n in rows:
        nxt=[]
        for p,g in parts:
            if np.all(n==0):nxt.append((p,g+[0.]));continue
            for sign,gate in ((1,1.),(-1,0.)):
                q=cut(p,sign*n)
                if q:nxt.append((q,g+[gate]))
        parts=nxt
    return [(p,np.asarray(g)) for p,g in parts]

def exact(model):
    wi,wh,wo=model;total=0.;leaves=0
    for a in range(3):
      for b in range(3):
       if a==b:continue
       k=3-a-b
       for s in(-1,1):
        N=np.stack([E[a]-E[b],E[b]-E[k],s*E[0]])
        B=np.linalg.inv(N);poly=[B[:,i]/np.linalg.norm(B[:,i]) for i in range(3)]
        g=(s*wi>0).astype(float);A=np.outer(wi,E[0])
        A2=wh@(g[:,None]*A)+np.outer(wi,E[1])
        for p2,g2 in split(poly,A2):
         A3=wh@(g2[:,None]*A2)+np.outer(wi,E[2])
         for p3,g3 in split(p2,A3):
          L=wo@(g3[:,None]*A3);q=p3
          for o in range(3):
           if o==b:continue
           n=L[b]-L[o]
           if np.all(n==0):
            if o<b:q=[];break
           else:q=cut(q,n)
           if not q:break
          total+=area(q);leaves+=1
    return total,leaves
