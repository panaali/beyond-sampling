import numpy as np
from feats4 import feats as f4
def feats(x):
    F=f4(x)
    F['g12n8']=F['g12']*(1-F['am8']); F['r(S9-x9)n8']=F['r(S9-x9)']*(1-F['am8'])
    F['am[2,6]']=F['am2']+F['am6']; F['rmo[2,6]']=F['r(x2-mo)']+F['r(x6-mo)']
    return F
