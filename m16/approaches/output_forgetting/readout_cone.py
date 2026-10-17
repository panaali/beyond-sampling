"""Exact interval-enclosed output certificate; all final-gate kink vertices."""
from fractions import Fraction as F

def clip(poly,A,B):
    out=[]
    for x,y in zip(poly,poly[1:]+poly[:1]):
        vx=A*x[0]+B*x[1];vy=A*y[0]+B*y[1]
        if vx>=0:out.append(x)
        if (vx<0<vy) or (vy<0<vx):
            t=vx/(vx-vy);out.append(tuple(x[i]+t*(y[i]-x[i]) for i in range(2)))
    return list(dict.fromkeys(out))

def thresholds(alpha,beta,a,c,wo,sign):
    alpha,beta,a,c=map(F,(alpha,beta,a,c));wo=[list(map(F,row)) for row in wo]
    axis=1 if sign=='+' else 0;j=max(range(len(wo)),key=lambda i:wo[i][axis]);roots=[]
    for l,row in enumerate(wo):
        d1,d2=wo[j][0]-row[0],wo[j][1]-row[1]
        risk=d1<0 if sign=='+' else d2<0
        if risk:
            denominator=-alpha*d1+beta*d2 if sign=='+' else alpha*d1-beta*d2
            assert denominator>0
            root=-alpha*(a*d1+c*d2)/denominator;roots.append((l,max(F(0),root)))
    k=max([F(0)]+[r for _,r in roots]);old=a if sign=='+' else c*alpha/beta
    assert 0<=k<=old
    return k,old,j,roots

def certificate(alpha,beta,a,c,wo,sign):
    alpha,beta,a,c=map(F,(alpha,beta,a,c));wo=[list(map(F,row)) for row in wo]
    k,old,j,roots=thresholds(alpha,beta,a,c,wo,sign)
    A,B=(a,-alpha) if sign=='+' else (c,-beta)
    rectangle=[(F(0),k),(alpha,k),(alpha,old),(F(0),old)]
    faces=[];failures=[]
    for active in (False,True):
        vertices=clip(rectangle,A if active else -A,B if active else -B);checks=[]
        for w,q in vertices:
            h=(a*w-alpha*q if active else F(0),c*w+beta*q) if sign=='+' else (a*w+alpha*q,c*w-beta*q if active else F(0))
            assert min(h)>=0
            for l,row in enumerate(wo):
                if l==j:continue
                margin=sum((wo[j][i]-row[i])*h[i] for i in range(2))
                wrongtie=margin==0 and l<j and q>k
                if margin<0 or wrongtie:failures.append({'w':str(w),'q':str(q),'competitor':l,'margin':str(margin),'wrongtie':wrongtie})
                checks.append({'w':str(w),'q':str(q),'competitor':l,'margin':str(margin),'excluded_lower_boundary':q==k})
        faces.append({'active':active,'vertices':[[str(w),str(q)] for w,q in vertices],'checks':checks})
    # A strictly positive axis for every q>old certifies the unbounded remainder.
    q=k/2;w=alpha
    h=(max(F(0),a*w-alpha*q),c*w+beta*q) if sign=='+' else (a*w+alpha*q,max(F(0),c*w-beta*q))
    scores=[sum(row[i]*h[i] for i in range(2)) for row in wo]
    actual=max(range(len(wo)),key=lambda i:scores[i])
    if k>0 and actual==j:raise AssertionError('positive threshold lacks endpoint falsifier')
    if failures:raise AssertionError(failures)
    return {'sign':sign,'threshold':str(k),'old_threshold':str(old),'winner':j,'roots':[(l,str(r)) for l,r in roots],'faces':faces,'below_threshold_falsifier':{'tested':k>0,'q':str(q),'w':str(w),'winner':actual},'uniform_interval_certificate':True,'largest_reachable_event_claim':False}
