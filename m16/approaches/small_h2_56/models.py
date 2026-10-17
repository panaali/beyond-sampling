import hashlib
from pathlib import Path
import numpy as np

def load(t,seed=0):
 import torch
 p=Path(f'/workspace/clean-neurons/work/checkpoints/H2_T{t}_seed{seed}.pth')
 sd=torch.load(p,weights_only=True,map_location='cpu')
 return (sd['rnn.weight_ih_l0'].numpy().astype(float).ravel(),sd['rnn.weight_hh_l0'].numpy().astype(float),sd['linear.weight'].numpy().astype(float)),hashlib.sha256(p.read_bytes()).hexdigest()
def noisy(model,sigma):
 r=np.random.default_rng(1);return tuple(w+r.normal(size=w.shape)*np.sqrt(np.mean(w*w))*sigma for w in model)
def sample(model,n=131072,seed=48001):
 wi,wh,wo=model;rng=np.random.default_rng(seed);good=0
 for start in range(0,n,65536):
  x=rng.normal(size=(min(n-start,65536),len(wo)));h=np.zeros((len(x),len(wi)))
  for k in range(len(wo)):h=np.maximum(0,x[:,k,None]*wi+h@wh.T)
  good+=int(((h@wo.T).argmax(axis=1)==np.argsort(x,axis=1)[:,-2]).sum())
 p=good/n;return dict(n=n,correct=good,estimate=p,se=float(np.sqrt(p*(1-p)/n)))
