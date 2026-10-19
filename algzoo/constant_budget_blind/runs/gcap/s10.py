from lib import *
n=30000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
G=np.c_[[np.maximum(Mm[:,t-1]-X[:,t],0) for t in range(1,10)]].T
g=np.c_[[np.maximum(Ss[:,t-1]-X[:,t],0) for t in range(2,10)]].T
U=np.c_[[np.maximum(X[:,t]-Mm[:,t-1],0) for t in range(1,10)]].T
u=np.c_[[np.maximum(X[:,t]-Ss[:,t-1],0) for t in range(2,10)]].T
big=np.c_[X,Mm[:,1:],Ss[:,1:],G,g]
big2=np.c_[big,U,u]
am=np.argmax(X,1); mx=X.max(1)
for i in range(8):
    R1=relufit(big,Hs[:,i])[0]; R2=relufit(big2,Hs[:,i])[0]
    # linear (no relu) on x as reference
    print(i,'ceil big',round(R1,3),'big+U',round(R2,3))
