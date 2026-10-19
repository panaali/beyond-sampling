import numpy as np, probe
rng=np.random.default_rng(7)
np.set_printoptions(precision=1,suppress=True,linewidth=220)
# max=3 at k, second=S at j, rest ~ -1 +-0.1, x9 = 0 unless k/j = 9
for Sv in [2.0,0.0]:
  print("S value",Sv)
  for k in range(9):
    rows=[]
    for j in range(9):
        if j==k: continue
        x=-1+0.1*rng.standard_normal((50,10)); x[:,9]=-0.5; x[:,k]=3; x[:,j]=Sv
        rows.append(probe.hidden(x).mean(0))
    h=np.mean(rows,0)
    print("k=%d"%k,"h3 %.1f h6 %.1f h8 %.1f h12 %.1f | dense h1 %.1f h2 %.1f"%(h[3],h[6],h[8],h[12],h[1],h[2]))
