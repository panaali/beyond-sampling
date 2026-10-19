import numpy as np, probe, json
from featlib import feature
from dic3 import fitrelu
CH={0:["M8","x8","x7","M9-x9","M4-x4"],1:["M9-x6","Mpo8","xn6","PO*g","g"],2:["M9-x9","PO*g","PE*g"],
3:["M7","r(x2-x4)","max(x1,x3)","max(x1,x5)","max(x0,x2)"],4:["M7","Mpe6","min(x4,x6)","max(x0,x6)","PE*g"],
5:["M8","min(x4,x7)","M9-x8","M9-x9","min(x5,x6)"],6:["M7-x7","Mpo6","PE*g"],7:["x4","M8","xn5","xn7","E8M8"],
8:["M8","cap8","PE*g","cap5","r(m2-x9)"],9:["M7","min(x4,x5)","max(x2,x6)","max(x4,x5)","max(x0,x2)"],
10:["r(x9-m2)"],11:["M9-x9","pe*g"],12:["r(x3-x1)","cap3","max(x0,x2)","r(x4-x5)","r(x1)"],
13:["M8","r(x8-M7)","max(x2,x4)","r(x0-x6)","min(x5,x7)"],14:["am6*x6","am6*x9","r(x6-m2)","am6*x3","am6*x7"],
15:["M5-x5","r(x2)","r(x6-M5)","r(x1-M0)","PE*g"]}
x=np.random.default_rng(41).standard_normal((60000,10)); h=probe.hidden(x)
xt=np.random.default_rng(99).standard_normal((100000,10)); ht=probe.hidden(xt)
out={}
for i,ks in CH.items():
    b=fitrelu(np.stack([feature(k,x) for k in ks],1),h[:,i],50)
    p=np.maximum(np.stack([feature(k,xt) for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    out[i]=(ks,[round(float(v),3) for v in b],float(r2)); print(i,'%.4f'%r2,ks,np.round(b,3))
json.dump(out,open('final.json','w'))
