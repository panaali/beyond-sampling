from fitlib import *
import numpy as np
def addset(name,S,kind):
    for FF in (F,Fv):
        FF[name]=sum(FF[kind%j] for j in S)
for i,j in [(2,3),(15,1),(19,6),(22,7),(14,8)]:
    ev(i,['r(x%d-to)'%j,'top2_%d'%j,'am%d'%j]); ev(i,['r(x%d-to)'%j,'top2_%d'%j,'am%d'%j,'r(x0-mo)'])
    ev(i,['r(x%d-so)'%j,'r(x%d-to)'%j,'top2_%d'%j,'am%d'%j])
for i,S in [(27,[2,6]),(11,[0,4,6]),(10,[4,5]),(26,[0,2,4])]:
    n=str(S); addset('am'+n,S,'am%d'); addset('rmo'+n,S,'r(x%d-mo)')
    ev(i,['am'+n,'rmo'+n])
