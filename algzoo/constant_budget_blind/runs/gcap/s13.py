from lib import *
n=40000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
G=np.c_[[np.maximum(Mm[:,t-1]-X[:,t],0) for t in range(1,10)]].T
g=np.c_[[np.maximum(Ss[:,t-1]-X[:,t],0) for t in range(2,10)]].T
P=[];pn=[]
for s in range(10):
    for t in range(s+1,10): P.append(np.maximum(X[:,s]-X[:,t],0)); pn.append((s,t))
P=np.array(P).T
A=np.abs(X)
for i in range(8):
    R1=relufit(np.c_[X,P],Hs[:,i])[0]; R2=relufit(np.c_[X,P,Mm[:,1:],Ss[:,1:],G,g],Hs[:,i])[0]; R3=relufit(np.c_[X,P,Mm[:,1:],Ss[:,1:],G,g,A],Hs[:,i])[0]
    print(i,'x+pairs',round(R1,3),'+M,S,G,g',round(R2,3),'+|x|',round(R3,3))
