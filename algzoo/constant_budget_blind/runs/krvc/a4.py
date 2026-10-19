import numpy as np, probe
rng=np.random.default_rng(2)
x=rng.standard_normal((20000,10)); H=probe.hidden(x)
# sensitivity of each neuron to each input position: mean |dh| when x_j resampled
S=np.zeros((32,10))
for j in range(10):
    x2=x.copy(); x2[:,j]=rng.standard_normal(len(x)); H2=probe.hidden(x2)
    S[:,j]=np.abs(H2-H).mean(0)/(H.std(0)+1e-9)
np.set_printoptions(linewidth=200)
for i in range(32): print(i,np.round(S[i],2))
