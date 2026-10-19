import numpy as np
x=np.load('x2.npy'); h=np.load('h2.npy')
y=h[:,21]
p=x[:,:9]; s=np.sort(p,1)[:,::-1]
def lin(X,nm):
    X1=np.c_[X,np.ones(len(y))]; b=np.linalg.lstsq(X1,y,rcond=None)[0]
    print(nm,'R2 %.3f'%(1-((X1@b-y)**2).mean()/y.var()),np.round(b,2))
lin(x,'x'); lin(s,'sorted prefix'); lin(np.c_[x,s],'x+sorted')
lin(np.c_[np.maximum(x,0)],'relu x'); lin(np.c_[x,np.maximum(x,0)],'x+relu x')
P=np.maximum.accumulate(x,1); lin(np.c_[x,P],'x+runmax')
lin(np.c_[np.maximum(x-P[:,[8]],-9)],'x-P8')
lin(np.c_[x,P,np.maximum(x[:,1:]-P[:,:-1],0)],'x+runmax+newmax')
