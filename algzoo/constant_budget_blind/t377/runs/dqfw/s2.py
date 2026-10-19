from lib import *
M8=rm[:,8]
for i in [7,16,21,23]:
    y=h[:,i]
    print(i, fit(y,[relu(x[:,9]-M8)]), fit(y,[relu(x[:,9]-M8),relu(x[:,9]-m2)*(am==9)]))
    # fraction positive when am!=9
    print('  pos when am!=9', (y[am!=9]>0).mean(), 'zero when am==9',(y[am==9]==0).mean())
