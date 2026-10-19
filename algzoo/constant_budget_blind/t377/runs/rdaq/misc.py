import numpy as np
from featlib import *
from feats import fit_relu, fit_gated, r2
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
M=topk(x,9,0);S=topk(x,9,1);T=topk(x,9,2)
def show(i,nm,X,g=None):
    X=np.column_stack(X)
    r,c=fit_relu(X,H[:,i],20) if g is None else fit_gated(X,H[:,i],g>0.5,20)
    print(i,nm,'%.3f'%r,np.round(c,2))
g14=inset(am9(x),[0,1,2,3,5])
show(14,'g:[M,S,I1_3]',[M,S,I1(x,3)],g14)
show(14,'g:[x3,M,S]',[x[:,3],M,S],g14)
show(14,'g:[M,S,I1_3,M6]',[M,S,I1(x,3),topk(x,6,0)],g14)
show(14,'g6:[M,S,I1_3]',[M,S,I1(x,3)],inset(am9(x),[0,1,2,3,4,5]))
show(14,'[M6-M9,M,S,I1_3]',[topk(x,6,0)-M,M,S,I1(x,3)])
s07=x[:,:7].sum(1); t8=-np.sort(-x[:,:8],1)
for nm,X in [('sum06,t8 top3',[s07,t8[:,0],t8[:,1],t8[:,2]]),('sum06,x7,M8',[s07,x[:,7],t8[:,0]]),('sum07,t8',[x[:,:8].sum(1),t8[:,0],t8[:,1],t8[:,2]])]:
    X=np.column_stack(X); print(21,nm,'lin %.3f'%r2(X,H[:,21])[0], 'relu %.3f'%fit_relu(X,H[:,21],10)[0])
show(30,'g J... [x8,So,x9]',[x[:,8],So9(x,8),x[:,9]],I1(x,8))
show(30,'g top2 [x8,So,x9]',[x[:,8],So9(x,8),x[:,9]],top2(x,8))
show(30,'g top2 [x8,So,x9,Mo]',[x[:,8],So9(x,8),x[:,9],Mo9(x,8)],top2(x,8))
show(17,'greedy',[rSo9(x,8),J1(x,6),I2(x,7),topk(x,5,0)])
show(11,'greedy',[rSo9(x,8),I1(x,2),rMo9(x,8),topk(x,6,0)])
