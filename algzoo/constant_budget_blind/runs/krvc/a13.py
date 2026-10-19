from fitlib import *
import numpy as np
for FF in (F,Fv):
    FF['g12n8']=FF['g12']*(1-FF['am8'])
    FF['r(S9-x9)n8']=FF['r(S9-x9)']*(1-FF['am8'])
ev(1,['g12n8']); ev(14,['r(so-x8)']); ev(14,['r(so-x8)','r(x8-to)']); ev(14,['r(so-x8)','r(x8-to)','r(x0-to)'])
for i in [0,13,29,6,18]:
    ev(i,['r(S9-x9)']); ev(i,['r(S9-x9)n8']); ev(i,['r(S9-x9)','am8'])
ev(9,['S9','T9','x9','g12'])
ev(17,['M9','S9','T9'],relu=False)
