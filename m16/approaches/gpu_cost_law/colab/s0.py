import sys, os, time, json
B="/content/cn/m16/approaches"; sys.path[:0]=[B+"/estT_fix",B+"/understand_m43",B+"/arc_m43_repro","/content/cn/m16"]
import estT; from m43 import Ops; from zoo44 import zoo
m=zoo(8,4,0); out={}
for mode in ["exact","recipe108"]:
    o=Ops(); t=time.time(); a=estT.estT(m,o,mode=mode); out[mode]=dict(acc=a,ops=o.n,sec=time.time()-t,**estT.STATS)
print(json.dumps(out))
