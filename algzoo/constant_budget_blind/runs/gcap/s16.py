from lib import *
n=40000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
def top(a,k): return -np.sort(-a,1)[:,k]
t3=lambda T: sum(top(X[:,:T+1],k) for k in range(3))/3
y=Hs[:,4]
def show(cols,nm):
    R,c,p=relufit(np.c_[cols].T,y); print('n4',nm,round(R,4),c.round(3))
gap8=np.maximum(t3(7)-X[:,8],0); gap9=np.maximum(t3(8)-X[:,9],0)
show([gap8],'gap8'); show([gap8,gap9],'gap8 gap9')
show([np.maximum(Mm[:,7]-X[:,8],0),np.maximum(Ss[:,7]-X[:,8],0),gap9],'G8 g8 gap9')
for a in [0.9,1.0,1.1,1.2,1.3]:
    gp=np.maximum(a*t3(7)-X[:,8],0); show([gp,gap9],f'gap8(a={a}) gap9')
