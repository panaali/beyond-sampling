"""Constructive counterexample for every four-word with nonzero last input."""
from fractions import Fraction as F

def axis_preimage(wi,wh,x1,x2,r):
    alpha,beta=-F(wi[0]),F(wi[1]);a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1])
    if not (min(alpha,beta,b,c,d)>0 and a>d and b*c>a*d):raise ValueError('model outside proved parameter regime')
    x1,x2,r=F(x1),F(x2),F(r)
    if r<0:raise ValueError('axis target must be nonnegative')
    delta=b*c-a*d;t=r-beta*x2
    thresholds=[F(0),-t/d,(a*t/c-alpha*x2)/(b-a*d/c),
                (d*t/c+(d*alpha+b*beta)*x1)/(b-d*d/c),
                (t+(c*alpha+a*beta)*x1)/(a-d)]
    v=max(thresholds)+1;u=(t+d*v)/c
    h0=((-d*(u+alpha*x1)+b*(v-beta*x1))/delta,
        (-c*(u+alpha*x1)+a*(v-beta*x1))/delta)
    if min(h0)<0 or min(u,v)<=0:raise AssertionError('constructive positivity')
    return h0,{'first_target':(u,v),'second_target':(F(0),r),'thresholds':thresholds}

def four_word_witness(wi,wh,word):
    if len(word)!=4:raise ValueError('four inputs required')
    x1,x2,x3,x4=map(F,word)
    if x4==0:raise ValueError('zero last input excluded from obstruction')
    alpha,beta=-F(wi[0]),F(wi[1]);b,d=-F(wh[0][1]),-F(wh[1][1])
    r=max(F(0),-alpha*x3/b,beta*x3/d)+1
    return axis_preimage(wi,wh,x1,x2,r)

def direct(wi,wh,initial,word):
    w1,w2=map(F,wi);aa,bb=map(F,wh[0]);cc,dd=map(F,wh[1]);p,q=initial;states=[]
    for x in map(F,word):
        p,q=max(aa*p+bb*q+w1*x,F(0)),max(cc*p+dd*q+w2*x,F(0));states.append((p,q))
    return states
