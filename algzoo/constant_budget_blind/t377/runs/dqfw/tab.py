from lib import *
import sys
for i in map(int,sys.argv[1:]):
    y=h[:,i]; print('neuron',i,'rows am, cols a2 (mean h)')
    for p in range(10):
        print(p,' '.join('%5.1f'%(y[(am==p)&(a2==q)].mean()) if p!=q else '    -' for q in range(10)))
