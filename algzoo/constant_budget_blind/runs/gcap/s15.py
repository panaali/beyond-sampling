from lib import *
n=40000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
def top(a,k): return -np.sort(-a,1)[:,k]
t3_8=sum(top(X[:,:9],k) for k in range(3))/3; t3_7=sum(top(X[:,:8],k) for k in range(3))/3
y=Hs[:,5]
for a in [1.0,1.05,1.08,1.1]:
    f=np.maximum(a*t3_8-X[:,9],0); c=(f@y)/(f@f); print('n5 a',a,'c',round(c,3),'R2',round(r2(c*f,y),4))
y=Hs[:,4]
def show(cols,nm):
    R,c,p=relufit(np.c_[cols].T,y); print('n4',nm,round(R,4),c.round(3))
show([t3_7,X[:,8],X[:,9]],'t3_7 x8 x9')
show([t3_7,X[:,8],np.maximum(t3_8-X[:,9],0)],'t3_7 x8 gap9(t3_8)')
show([t3_7,X[:,8],np.maximum(Mm[:,8]-X[:,9],0)],'t3_7 x8 G9')
show([t3_7,X[:,8],np.maximum(Mm[:,8]-X[:,9],0),np.maximum(Ss[:,8]-X[:,9],0)],'t3_7 x8 G9 g9')
show([t3_7,X[:,8],X[:,9],np.maximum(t3_8-X[:,9],0)],'t3_7 x8 x9 gap9')
show([Mm[:,7],Ss[:,7],X[:,8],np.maximum(t3_8-X[:,9],0)],'M7 S7 x8 gap9')
