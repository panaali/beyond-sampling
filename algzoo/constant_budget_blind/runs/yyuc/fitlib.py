import numpy as np
from scipy.optimize import least_squares
def feats(x):
  P=-np.sort(-x[:,:9],1); pa=np.argmax(x[:,:9],1)
  o=np.argsort(-x[:,:9],1)
  d=dict(P1=P[:,0],P2=P[:,1],P3=P[:,2],P4=P[:,3],x9=x[:,9],x8=x[:,8],pa=pa,pb=o[:,1])
  s=-np.sort(-x,1); d.update(s1=s[:,0],s2=s[:,1],s3=s[:,2]); oo=np.argsort(-x,1); d.update(am=oo[:,0],sm=oo[:,1])
  d['M7']=x[:,:8].max(1)
  return d
def relufit(F,y,c0=None):
  if c0 is None:
    c0,*_=np.linalg.lstsq(F,y,rcond=None)
  r=least_squares(lambda c: np.maximum(F@c,0)-y, c0)
  c=r.x; pred=np.maximum(F@c,0)
  return c,1-((pred-y)**2).mean()/y.var(),pred
