from fractions import Fraction as F

def diagonal(signs):return [[signs[i] if i==j else 0 for j in range(4)] for i in range(4)]

def cases():
    p=F(1,16)
    valid=[('positive',diagonal([1]*4),15,F(1)),('negative',diagonal([-1]*4),0,F(1)),('sole_first',diagonal([1,-1,-1,-1]),1,F(1)),('boundary',[[1,0,0,0],[1,-1,0,0],[0,0,-1,0],[0,0,0,-1]],1,F(1,2))]
    guards=[('mixed',[[1,0,0,0],[-1,1,0,0],[0,0,1,0],[0,0,0,1]],'mixed coordinate signs'),('zero_axis',[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,1,1,0]],'zero coordinate span'),('singular',[[1,1,1,1]]*4,'singular simplex'),('shape',[[1,0,0,0]]*3,'four by four rays')]
    laws=[('zero',F(0),F(0),F(0),F(0)),('all',p,p,F(0),F(0)),('low',p/4,p/3,p/4,p/3),('high',2*p/3,3*p/4,p/4,p/3),('cross_half',p/4,3*p/4,p/4,p/2),('half',p/2,p/2,p/2,p/2)]
    invalid=[('inverted',p,F(0)),('negative',-p,F(0)),('above',F(0),2*p)]
    return valid,guards,laws,invalid
