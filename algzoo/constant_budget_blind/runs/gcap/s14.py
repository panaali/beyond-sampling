from lib import *
n=40000; X=x[:n]; Hs=H[:n]
def top(a,k): return -np.sort(-a,1)[:,k]
p8=X[:,:9]; p7=X[:,:8]
T={k:top(p8,k) for k in range(5)}; T7={k:top(p7,k) for k in range(5)}
def show(i,cols,names):
    R,c,p=relufit(np.c_[cols].T,Hs[:,i]); print(i,names,round(R,4),c.round(3))
show(5,[T[0],T[1],X[:,9]],'M8 S8 x9')
show(5,[T[0],T[1],T[2],X[:,9]],'top0-2 x9')
show(5,[T[0],T[1],T[2],T[3],X[:,9]],'top0-3 x9')
show(5,[T[0]+T[1]+T[2],X[:,9]],'sumtop3 x9')
show(5,[T[0]+T[1],X[:,9]],'sumtop2 x9')
show(4,[T7[0],T7[1],X[:,8],X[:,9]],'M7 S7 x8 x9')
show(4,[T7[0],T7[1],T7[2],X[:,8],X[:,9]],'top0-2(x0..7) x8 x9')
show(4,[T7[0]+T7[1]+T7[2],X[:,8],X[:,9]],'sumtop3 x8 x9')
show(4,[T7[0]+T7[1],X[:,8],X[:,9]],'sumtop2 x8 x9')
show(4,[T7[0]+T7[1],X[:,8],np.maximum(T[1]-X[:,9],0)],'sumtop2 x8 g9')
show(4,[T7[0]+T7[1]+T7[2],X[:,8],np.maximum(T[1]-X[:,9],0)],'sumtop3 x8 g9')
