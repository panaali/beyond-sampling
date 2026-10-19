import numpy as np, probe, labels
x=np.random.default_rng(99).standard_normal((50000,10));h=probe.hidden(x)
r=[]
for i in range(16):
    p=labels.f(i,x);r.append(1-((p-h[:,i])**2).mean()/h[:,i].var())
    print(i,labels.LABELS[i][0],labels.CONSTANTS[i],'R2 %.3f'%r[-1])
print('mean R2 %.3f'%np.mean(r))
