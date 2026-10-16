import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=4096
for name,fn,tr in [('23',faces23,0.029203),('12',faces12,0.0092235)]:
    fc=fn(n); Z=fc['Z']
    for rad in [True]:
        F,R,C=F_faces(M,fc,60,radial=rad); F*=Z
        print(name,'est',F.mean(),'truth',tr,'sd',F.std(),'cost(passes)',C.mean(),'var*cost',F.var()*C.mean())
