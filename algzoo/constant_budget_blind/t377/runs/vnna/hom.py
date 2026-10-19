import numpy as np, probe
np.set_printoptions(precision=3,suppress=True,linewidth=200)
x=np.random.default_rng(15).standard_normal((5000,10)); H=probe.hidden(x); H2=probe.hidden(2*x)
print(np.abs(H2-2*H).max(0))
G=np.load('G.npy'); gx=np.load('Gx.npy'); Hg=probe.hidden(gx)
E=(G*gx[:,:,None]).sum(1)
err=np.abs(E-Hg)
print("median Euler err",np.median(err,0)); print("frac err>0.1",(err>0.1).mean(0))
print("max |grad| median", np.median(np.abs(G).max(1),0))
