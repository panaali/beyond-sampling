import numpy as np, probe
d=np.load('grad.npz'); x,H,G=d['x'],d['H'],d['G']
r=(G*x[:,:,None]).sum(1)-H
print(np.percentile(np.abs(r),[50,90,99,99.9]))
print(H.dtype, probe.hidden(x[:5]).dtype)
H2=probe.hidden(2*x[:2000]); print("scale err",np.abs(H2-2*H[:2000]).max())
print("G sample", np.abs(G).max())
