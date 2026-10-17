"""Exact binary-weight witness for divergence of top-two normalized residual variance."""
import argparse
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth')
a=p.parse_args(); path=Path(a.checkpoint)
s=torch.load(path,map_location='cpu',weights_only=True)
wi=[Q(float(v)) for v in s['rnn.weight_ih_l0'][:,0]]
wh=[[Q(float(v)) for v in row] for row in s['rnn.weight_hh_l0']]
wo=[[Q(float(v)) for v in row] for row in s['linear.weight']]
h=[Q(0)]*len(wi); x=[0,0]+[-1]*8
for u in x:
    h=[max(Q(0),sum(v*z for v,z in zip(row,h))+a*u) for row,a in zip(wh,wi)]
f=[sum(v*z for v,z in zip(row,h)) for row in wo]
assert f[0]!=f[1]
print('checkpoint_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
print('witness',x)
print('exact_f0_minus_f1',f[0]-f[1])
print('rounded_f0_minus_f1',float(f[0]-f[1]))
print('centered_logit_norm',float(sum((v-sum(f)/len(f))**2 for v in f))**.5)
