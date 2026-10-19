import numpy as np, probe
from feat import r2; from fit import fitrelu
from lib2 import lib3
np.set_printoptions(precision=3,suppress=True,linewidth=160)
rng=np.random.default_rng(19); N=40000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
xt=rng.standard_normal((100000,10)); ht=probe.hidden(xt)
L=lib3(x); Lt=lib3(xt)
L['DR45']=L['DR4']+L['DR5']; Lt['DR45']=Lt['DR4']+Lt['DR5']
L['DA45']=L['DA4']+L['DA5']; Lt['DA45']=Lt['DA4']+Lt['DA5']
specs={0:[['DR45','DR3','x8'],['DR4','DR5','DR3','x8'],['DA4','DA5','DA3','x8'],['DR45','DR3','x8','M+S']],
1:[['DR6','DR4','m1_9','MSlag'],['DA6','DA4','m1_9','MSlag'],['DR6','DR4','M+S'],['DA6','DA4','M+S']],
2:[['m1_9','DR3','DR5','DR6'],['m1_9','DA3','DA5','DA6'],['M+S','DR3','DR5'],['M+S','DA3','DA5']],
3:[['m1_9','DR3','DR5','DA7'],['m1_9','DR3','DR5','DR7'],['m1_9','DA3','DA5','x7'],['m1_9','DR3','DR5','x7'],['m1_9','DA3','DA5','DA7']],
4:[['m1_8','m2_8','x8','x9'],['DA8','x9','relusum'],['DR8','DR9'],['DA8','DA9']],
5:[['M+S','x9','relusum'],['DA9','relusum'],['DR9'],['DA9'],['m1_9','m2_9','x9','relusum']],
6:[['M+S','DR4','DR2','x8'],['M+S','DA4','DA2','x8'],['M+S','DR4','DR2'],['MSlag','DR4','DR2','x8']],
7:[['MSlag','DA5','DR6','x9'],['MSlag','DR5','DR6','x9'],['MSlag','DA5','DA6','x9'],['M+S','DR5','DR6','x9']]}
one=np.ones(N); onet=np.ones(len(xt))
for i,ss in specs.items():
    for sel in ss:
        A=np.c_[np.array([L[q] for q in sel]).T,one]; c,_=fitrelu(A,h[:,i])
        At=np.c_[np.array([Lt[q] for q in sel]).T,onet]
        print(i,sel,'test',round(r2(ht[:,i],np.maximum(At@c,0)),3),np.round(c,3))
