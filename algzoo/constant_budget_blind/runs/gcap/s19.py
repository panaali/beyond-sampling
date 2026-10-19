import numpy as np
import labels_core as L
from lib import relufit
x=np.load('x.npy')[:40000];H=np.load('H.npy')[:40000]
tries={6:[['LEVEL','G4','D24','g4','D28'],['LEVEL','G4','g4','D24','x8'],['LEVEL','x4','D24','x8','g2'],['LEVEL','gap4','x4','D24','x8']],
0:[['LEVEL','x4','x5','x8','g3'],['LEVEL','x4','x5','x8','G3'],['LEVEL','G5','G4','x8','x3'],['LEVEL','x4','x5','x8','D35'],['LEVEL','gap4','gap5','x8','x3']],
2:[['LEVEL','G5','g5','D35','D89'],['LEVEL','x5','x3','D35','D89'],['LEVEL','G5','D35','g6','D89']],
7:[['LEVEL','G5','D35','x9','g6'],['LEVEL','x5','x6','x9','D35']]}
for i,tl in tries.items():
    for names in tl:
        F=np.c_[[L.feat(x,n) for n in names]].T; R,c,p=relufit(F,H[:,i],it=12); print(i,names,round(R,4),c.round(3))
