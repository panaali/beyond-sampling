from lib import *
one=np.ones(len(x))
cands={5:[['M8','x9'],['M8','x9','S8'],['M9','x9'],['M8','x9','S8','x8']],
       4:[['M7','x8'],['M7','x8','x9'],['M7','x8','x9','S7'],['M8','x8','x9','S8'],['M7','x8','x9','M9']]}
def feat(n):
    k=n[0];t=int(n[1:]); return {'x':x,'M':M,'S':S}[k][:,t]
for i,cl in cands.items():
    for c in cl:
        F=np.c_[[feat(n) for n in c]].T
        R,co,p=relufit(F,H[:,i]); print(i,c,round(R,4),co.round(3))
