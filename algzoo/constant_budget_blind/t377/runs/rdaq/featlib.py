import numpy as np
def relu(z): return np.maximum(z,0)
def topk(x,L,k):
    """k-th largest (0=largest) among x0..x_{L-1}"""
    return -np.sort(-x[:,:L],1)[:,k]
def oth9(x,j):
    o=np.delete(x[:,:9],j,1) if j<9 else x[:,:9]
    return -np.sort(-o,1)
def Mo9(x,j): return oth9(x,j)[:,0]
def So9(x,j): return oth9(x,j)[:,1]
def am9(x): return np.argmax(x[:,:9],1)
def a2_9(x): return np.argsort(-x[:,:9],1)[:,1]
def I1(x,j): return (am9(x)==j)*1.0
def I2(x,j): return (a2_9(x)==j)*1.0
def top2(x,j): return I1(x,j)+I2(x,j)
def J1(x,j): return (np.argmax(x,1)==j)*1.0
def so(x,j): return x[:,:9].sum(1)-(x[:,j] if j<9 else 0)
def inset(v,S): return np.isin(v,S)*1.0
def rSo9(x,j): return relu(x[:,j]-So9(x,j))
def rMo9(x,j): return relu(x[:,j]-Mo9(x,j))
def amax(x): return np.argmax(x,1)
def asec(x): return np.argsort(-x,1)[:,1]
def s_all(x,k): return -np.sort(-x,1)[:,k]
def pair30(x):
    a=amax(x); b=asec(x); P=[3,5,6,7]
    return (((a==8)&np.isin(b,P))|((b==8)&np.isin(a,P)))*1.0
