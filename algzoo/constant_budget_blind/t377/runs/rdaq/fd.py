import numpy as np, probe
rng=np.random.default_rng(2)
x=rng.standard_normal((20000,10))
np.set_printoptions(precision=3,suppress=True,linewidth=220)
H0=np.asarray(probe.hidden(x))
G=[]
for j in range(10):
    x2=x.copy(); x2[:,j]+=1e-3
    H1=np.asarray(probe.hidden(x2))
    g=(H1-H0)/1e-3
    G.append(g)
G=np.stack(G,2) # N,32,10
# x9 slope where active
for i in range(32):
    a=H0[:,i]>1e-4
    g9=G[a,i,9]
    print(i,'x9 slope uniq',np.unique(np.round(g9,3))[:6], ' median grad',np.median(G[a,i,:],0))
np.save('G2.npy',G); np.save('x2.npy',x); np.save('H2.npy',H0)
