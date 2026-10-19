import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=220)
# bg -2, x9=-2; x_p=1, x_q=u ; neurons 0,3,6
us=np.array([-1.5,-1,-0.5,0,0.5,0.9])
for p in [0,1,3,4]:
  for q in range(9):
    if q==p: continue
    X=np.full((len(us),10),-2.0); X[:,p]=1; X[:,q]=us
    Hh=probe.hidden(X)
    print('p',p,'q',q,' n0',Hh[:,0],' n3',Hh[:,3],' n6',Hh[:,6])
