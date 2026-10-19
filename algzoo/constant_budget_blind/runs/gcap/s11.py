from lib import *
np.set_printoptions(precision=2,suppress=True,linewidth=160)
n=30000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
G=np.c_[[np.maximum(Mm[:,t-1]-X[:,t],0) for t in range(1,10)]].T
g=np.c_[[np.maximum(Ss[:,t-1]-X[:,t],0) for t in range(2,10)]].T
big=np.c_[X,Mm[:,1:],Ss[:,1:],G,g]
o=np.argsort(-X,1); am=o[:,0]; se=o[:,1]
for i in [0,1,2,3,6,7]:
    R,c,p=relufit(big,Hs[:,i]); res=Hs[:,i]-p
    print(i,'resid mean by argmax',np.array([res[am==t].mean() for t in range(10)]))
    print('   resid by argmax std', np.array([res[am==t].std() for t in range(10)]),'total std',res.std().round(2))
    print('   mean h by argmax',np.array([Hs[am==t,i].mean() for t in range(10)]))
