import numpy as np, probe
np.set_printoptions(linewidth=200,suppress=True)
x=np.random.default_rng(5).standard_normal((100000,10))
h=probe.hidden(x); am=np.argmax(x,1)
for i in [14,12,15,3,9,13]:
    y=h[:,i]; print('neuron',i)
    for t in range(10):
        m=am==t; b=np.linalg.lstsq(x[m],y[m],rcond=None)[0]; r2=1-((y[m]-x[m]@b)**2).mean()/y[m].var()
        print(' am',t,'mean %.1f frac+ %.2f r2 %.2f'%(y[m].mean(),(y[m]>0).mean(),r2),b.round(1))
