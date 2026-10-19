from lib import *
n=30000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
G=np.c_[[np.maximum(Mm[:,t-1]-X[:,t],0) for t in range(1,10)]].T
g=np.c_[[np.maximum(Ss[:,t-1]-X[:,t],0) for t in range(2,10)]].T
big=np.c_[X,Mm[:,1:],Ss[:,1:],G,g]
sx=-np.sort(-X,1)
F={}
for t in range(10): F[f'|x{t}|']=abs(X[:,t]); F[f'n{t}']=np.maximum(-X[:,t],0)
for k in range(10): F[f'sorted{k}']=sx[:,k]
F['min']=X.min(1); F['negsum']=np.maximum(-X,0).sum(1); F['records']=(np.diff(Mm,axis=1)>0).sum(1)
Rm=np.minimum.accumulate(X,1)
for t in range(1,10): F[f'runmin{t}']=Rm[:,t]
for t in range(1,10): F[f'drop{t}']=np.maximum(X[:,t-1]-X[:,t],0); F[f'rise{t}']=np.maximum(X[:,t]-X[:,t-1],0)
for i in [0,1,2,3,6,7]:
    R,c,p=relufit(big,Hs[:,i]); res=Hs[:,i]-p
    cc={k:np.corrcoef(v,res)[0,1] for k,v in F.items()}
    print(i,[(k,round(v,2)) for k,v in sorted(cc.items(),key=lambda a:-abs(a[1]))[:8]])
