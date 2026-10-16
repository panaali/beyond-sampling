import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=3600
for name,fn in [('23',faces23),('12',faces12)]:
    fc=fn(n); Z=fc['Z']
    # mass check (model-free): full segment weight
    for rad in [False,True]:
        F,R,C=F_faces(M,fc,60,radial=rad)
        print(name,'radial',rad,'e',Z*F.mean(),'sd',Z*F.std(),'cells',C.mean())
