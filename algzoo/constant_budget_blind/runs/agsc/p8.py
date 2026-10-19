import numpy as np
from traj import traj
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(7); N=2000
for sc in [0.0,0.5]:
  for V in [2.0]:
    x=rng.standard_normal((N,10))*sc; x[:,1]=V
    H=traj(x).mean(0)
    print('bg',sc,'V',V,'spike at x1; rows=neuron, cols=t=0..10'); print(H.T)
x=rng.standard_normal((N,10))*0.5; x[:,1]=2; x[:,5]=1.5
print('spike 2 at x1, 1.5 at x5'); print(traj(x).mean(0).T)
