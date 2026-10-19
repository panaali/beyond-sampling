from lib import *
n=40000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
EM=Mm[:,1:].mean(1)*10
def tm(cols,k): s=-np.sort(-X[:,cols],1); return s[:,:k].mean(1)
def gp(cols,t,k=1): return np.maximum(tm(cols,k)-X[:,t],0)
def show(i,cols,nm):
    R,c,p=relufit(np.c_[cols].T,Hs[:,i]); print(i,nm,round(R,4),c.round(3))
y=Hs[:,1]
for ex in [[],[4],[4,3],[4,2]]:
    cols=[c for c in range(6) if c not in ex]
    for k in [1,2,3]:
        show(1,[EM,gp(cols,6,k)],f'EM gap6 ref excl{ex} top{k}')
show(1,[EM,gp([0,1,2,3,5],6,1),gp([0,1,2,3,5],6,2)],'EM G6ex4 g6ex4')
show(1,[EM,gp([0,1,2,3,5],6,1),gp([0,1,2,3,5],6,2),X[:,4]],'EM G6ex4 g6ex4 x4')
show(1,[EM,gp([0,1,2,3,5],6,1),gp([0,1,2,3,5],6,2),X[:,4],X[:,8]],'EM G6ex4 g6ex4 x4 x8')
