import numpy as np
x=np.load('x2.npy');h=np.load('h2.npy')
p=x[:,:9]; s=np.sort(p,1)[:,::-1]; am=np.argmax(p,1)
M,S,T=s[:,0],s[:,1],s[:,2]; x9=x[:,9]
amf=np.argmax(x,1)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
for i in range(16):
    pred=np.zeros(len(x)); A=[];B=[];G=[];C=[]
    for j in range(9):
        g=(am==j)&(h[:,i]>0.3)
        X=np.c_[np.ones(g.sum()),M[g],S[g],x9[g]]
        c,*_=np.linalg.lstsq(X,h[g,i],rcond=None)
        gg=am==j
        pred[gg]=np.maximum(np.c_[np.ones(gg.sum()),M[gg],S[gg],x9[gg]]@c,0)
        C.append(c[0]);A.append(c[1]);B.append(c[2]);G.append(c[3])
    r2=1-((h[:,i]-pred)**2).mean()/h[:,i].var()
    print(i,'R2 %.3f'%r2); print('  a',np.array(A)); print('  b',np.array(B)); print('  g',np.array(G)); print('  c',np.array(C))
