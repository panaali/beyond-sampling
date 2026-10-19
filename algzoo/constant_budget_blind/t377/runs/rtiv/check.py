import numpy as np, probe, labels
x=np.random.default_rng(11).standard_normal((200000,10));H=probe.hidden(x)
for i in range(16):
    p=labels.f(i,x);r=1-((H[:,i]-p)**2).mean()/H[:,i].var()
    print(i,labels.LABELS[i][0],labels.CONSTANTS[i],"R2 %.3f"%r,len(labels.LABELS[i][1].split()),"words")
src=open('labels.py').read()
print([b for b in ["torch","alg_zoo","weight_hh","weight_ih","state_dict",".pth","probe"] if b in src])
