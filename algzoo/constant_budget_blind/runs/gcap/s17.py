from lib import *
np.set_printoptions(precision=2,suppress=True,linewidth=160)
n=40000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
sx=-np.sort(-X,1)
EM=Mm[:,1:].mean(1)*10
part={0:[4,5,8],1:[6,4],2:[3,5],3:[5,7,3],6:[4,2,8],7:[5,6,9]}
for i,P in part.items():
    y=Hs[:,i]; xp=X[:,P]
    a=relufit(np.c_[EM,xp],y); b=relufit(np.c_[Mm[:,1:],xp],y); c=relufit(np.c_[sx,xp],y); d=relufit(np.c_[Mm[:,1:],Ss[:,1:],xp],y)
    print(i,P,'EM',round(a[0],3),'M_t',round(b[0],3),b[1][:9],'sorted',round(c[0],3),c[1][:10],'M+S',round(d[0],3))
