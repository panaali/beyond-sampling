import numpy as np
from feats3 import feats as f3
from feats import feats as f1
def feats(x):
    F=f3(x); G=f1(x)
    for k,v in G.items():
        if k not in F: F['A_'+k]=v
    return F
