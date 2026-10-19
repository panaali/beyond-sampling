import numpy as np
x=np.load('x0.npy')[:15000];h=np.load('h0.npy')[:15000]
W={}
for a in range(10):
    for b in range(a,10):
        W[(a,b)]=x[:,a:b+1].max(1)
for i in range(16):
    y=h[:,i]; r=sorted([(np.corrcoef(y,v)[0,1],k) for k,v in W.items()],reverse=True)
    # also best window max minus another window max
    best=(0,)
    for k1,v1 in W.items():
        for k2,v2 in W.items():
            if k1==k2: continue
            c=np.corrcoef(y,np.maximum(v1-v2,0))[0,1]
            if c>best[0]: best=(c,k1,k2)
    print(i,[(round(c,2),k) for c,k in r[:3]],'relu(win1-win2):',round(best[0],2),best[1:])
