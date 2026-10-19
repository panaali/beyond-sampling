from lib import *
x9=x[:,9]; M8=rm[:,8]; S8=S[:,8]
out=[]
for i,g8 in [(0,1),(13,1),(29,1),(6,0),(18,0)]:
    g=(am!=8)*1. if g8 else np.ones(N)
    bf=lambda sl:[S8[sl]*g[sl],M8[sl]*g[sl],x9[sl]*g[sl]]
    mf=lambda ip,sl:[S8[sl]*g[sl]*ip,M8[sl]*g[sl]*ip]
    P,c,r=psearch(h[:,i],bf,mf,mode='relu'); print(i,P,np.round(c,3),'%.4f'%r,flush=True)
