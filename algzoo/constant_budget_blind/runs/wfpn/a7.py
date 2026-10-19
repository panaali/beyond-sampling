import numpy as np, probe
np.set_printoptions(suppress=True,linewidth=160)
rng=np.random.default_rng(2); N=20000
x=rng.standard_normal((N,10))
H=[]
for L in range(1,11):
  xp=np.zeros((N,10)); xp[:,10-L:]=x[:,:L]; H.append(probe.hidden(xp))
H=np.array(H) # H[L-1] = state after reading x0..x_{L-1}
np.save('Hpre.npy',H); np.save('xpre.npy',x)
print("mean by L\n",H.mean(1).round(2))
print("frac active by L\n",(H>0).mean(1).round(2))
for L in range(1,11):
  xx=x[:,:L]; M=xx.max(1); 
  srt=-np.sort(-xx,1)
  sec=srt[:,1] if L>1 else np.zeros(N)
  out=[]
  for i in range(8):
    y=H[L-1,:,i]
    out.append([round(np.corrcoef(y,v)[0,1],2) if y.std()>0 else 0 for v in [M,sec,xx[:,-1]]])
  print(L,out)
