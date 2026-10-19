import numpy as np
x=np.load('x1.npy')[:30000]; h=np.load('h1.npy')[:30000]
srt=-np.sort(-x,1); am=np.argmax(x,1); m=srt[:,0]; s1=srt[:,1]; g2=m-s1
M=np.maximum.accumulate(x,1); T=m[:,None]-x; G9=np.maximum(x[:,9]-M[:,8],0)
S8=-np.sort(-x[:,:9],1)[:,1]; Q9=np.maximum(S8-x[:,9],0)
I=np.eye(10)[am]
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
subsets=[[j for j in range(10) if b>>j&1] for b in range(1,1024)]
for i,f in [(3,T[:,9]),(6,T[:,9]),(12,T[:,9]),(8,T[:,7]),(14,T[:,4])]:
    y=h[:,i]
    for extra_name,extra in [('none',[]),('Q9',[Q9]),('G9',[G9])]:
        best=None
        for S in subsets:
            ind=I[:,S].sum(1)
            B=np.stack([f,g2*ind]+extra,1); r,c=fit(B,y)
            if best is None or r>best[0]: best=(r,S,c)
        # second set given first
        ind1=I[:,best[1]].sum(1); best2=None
        for S in subsets:
            ind=I[:,S].sum(1); B=np.stack([f,g2*ind1,g2*ind]+extra,1); r,c=fit(B,y)
            if best2 is None or r>best2[0]: best2=(r,S,c)
        print(i,extra_name,'1set %.3f'%best[0],best[1],best[2].round(2),'| 2set %.3f'%best2[0],best2[1],best2[2].round(2))
