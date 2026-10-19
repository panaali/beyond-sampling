import numpy as np
def mx(x,a,b): return x[:,a:b+1].max(1)
def sc(x,a,b): return np.sort(x[:,a:b+1],1)[:,-2]
def pre(x):
    P=x[:,:9]; s=np.sort(P,1); M8=s[:,-1]; s8=s[:,-2]; p8=P.argmax(1); return M8,s8,M8-s8,p8
def design(i,x):
    one=np.ones(len(x)); M8,s8,D,p8=pre(x); x9=x[:,9]; x8=x[:,8]
    if i==0: return [D,D*np.cos(0.8*(p8+1)),s8,sc(x,1,7),one]
    if i==1: return [M8,D*np.cos(0.95*(p8-6.5)),x9,one]
    if i==2: return [M8,mx(x,4,8),one]
    if i==3: return [mx(x,1,8),mx(x,0,1),x[:,6],mx(x,5,7),one]
    if i==4: return [x9,M8,one]
    if i==5: return [x8,sc(x,2,7),sc(x,0,5),one]
    if i==6: return [mx(x,0,7),mx(x,3,6),x9,one]
    if i==7: return [mx(x,0,9),mx(x,5,9),x9,one]
    if i==8: return [M8,D*np.cos(0.6*(p8-1.25)),D,x9,one]
    if i==9: return [x8,mx(x,2,8),sc(x,0,5),one]
    if i==10: return [M8,mx(x,1,4),mx(x,0,5),mx(x,2,3),one]
    if i==11: return [M8,mx(x,2,5),mx(x,3,6),mx(x,1,4),one]
    if i==12: return [M8,mx(x,0,3),sc(x,0,6),mx(x,3,6),one]
    if i==13: return [M8,mx(x,4,5),mx(x,0,1),sc(x,2,7),one]
    if i==14: return [M8,D*np.cos(0.8*(p8-3)),sc(x,0,7),mx(x,0,1),one]
    if i==15: return [mx(x,0,4),mx(x,3,4),mx(x,6,7),mx(x,0,3),one]
