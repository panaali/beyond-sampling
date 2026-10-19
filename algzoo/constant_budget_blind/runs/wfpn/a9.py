import numpy as np
np.set_printoptions(suppress=True,linewidth=200,precision=2)
H=np.load('Hpre.npy'); x=np.load('xpre.npy'); N=x.shape[0]
hf=H[9]
def r2(B,y):
  c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((B@c-y)**2).mean()/y.var(), c
for grp in [[5],[4,5],[3,4,5],[2,3,4,5]]:
  B=np.concatenate([H[:,:,g].T for g in grp],1)  # N x 10*len
  B=np.c_[B,x]
  print(grp,[round(r2(B,hf[:,i])[0],3) for i in range(8)])
B=np.c_[H[:,:,5].T, x]
for i in [0,1,2,6,7]:
  print(i, "coef on n5 at L=1..10:",r2(B,hf[:,i])[1][:10], "x:",r2(B,hf[:,i])[1][10:])
# n5 per step: fit relu(a*top4(prefix before) - b x_t)
for L in range(2,11):
  prev=x[:,:L-1]; k=min(4,L-1); top=(-np.sort(-prev,1))[:,:k].sum(1)
  y=H[L-1,:,5]; Bm=np.c_[top,x[:,L-1]]; m=y>0
  c,*_=np.linalg.lstsq(Bm[m],y[m],rcond=None)
  pr=np.maximum(Bm@c,0); print(L, c.round(3), round(1-((pr-y)**2).mean()/y.var(),3))
