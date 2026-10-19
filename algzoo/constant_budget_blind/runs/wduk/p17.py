import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=170)
print('h0',probe.hidden(np.zeros((1,10))))
for b in (0.5,0.9):
  X=[];idx=[]
  for i in range(10):
    for j in range(10):
      x=np.zeros(10);x[i]=1;
      if j!=i: x[j]=b
      X.append(x)
  H=probe.hidden(np.array(X))
  for n in (2,9,13):
    print('neuron',n,'b',b,'rows=max pos, cols=second pos (diag: alone)');print(H[:,n].reshape(10,10))
