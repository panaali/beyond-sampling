import numpy as np, json
exec("import numpy as np\n"+open('featlib_src.py').read())
from scipy.optimize import least_squares
SPEC={0:['x9','P1@01234567','P1@14578','P1@01234678','max01234578'],
1:['M7','max01347','max047','max0134678','2nd01345678'],
2:['2nd0123467','x2','P2@01345678','max14','P3@2'],
3:['x9','P1@01234567','P1@0367','max01234678','P2'],
4:['x9','P1@01234567','P2','max013467','P2@01234567'],
5:['max0235678','P1@5','P1@023568','P1@01345678','P2@5'],
6:['x9','P1@01234567','P1@23568','max01235678','P2'],
7:['x9','M7','max036','max0235678','P1@36'],
8:['max01347','P1@47','P1@01345678','2nd01234567','P1@01245678'],
9:['max0147','x1','P1@01347','P1@124578','P2@1'],
10:['x9','P2','P3'],
11:['x9','P1@123568','P1@02458','P2@01234567','P2@7'],
12:['x9','P1','P2','max25','max12568'],
13:['max01347','2nd01347','max47','x2','P2@47'],
14:['max124578','max0123568','max0236','P1@124578'],
15:['x9','P1']}
x=np.load('x1.npy'); H=np.load('H1.npy'); xt=np.load('x7.npy'); Ht=np.load('H7.npy')
C={}
for i,fs in SPEC.items():
  F=np.stack([_feat(f,x) for f in fs],1); y=H[:,i]
  c0,*_=np.linalg.lstsq(F[y>0],y[y>0],rcond=None)
  c=least_squares(lambda c: np.maximum(F@c,0)-y,c0).x
  Ft=np.stack([_feat(f,xt) for f in fs],1); yt=Ht[:,i]
  r2=1-((np.maximum(Ft@c,0)-yt)**2).mean()/yt.var()
  C[i]=[round(float(v),3) for v in c]
  print(i,fs,C[i],'fresh R2 %.4f'%r2)
json.dump({'SPEC':SPEC,'C':C},open('fit.json','w'))
