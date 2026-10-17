from pathlib import Path
import sys,types,hashlib,importlib.util
import numpy as np
ROOT=Path('/workspace/clean-neurons');HERE=Path(__file__).parent
for p in [ROOT/'m16/approaches/arc_m43_repro',ROOT/'m16/approaches/understand_m43',ROOT/'m16/approaches/understand_ladder',ROOT/'review/understand_ladder',ROOT/'review/understand_m43_t75']:
 sys.path.insert(0,str(p))
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
E75=module(ROOT/'m16/approaches/ladder_training_axis/est75_e37.py','reviewed_e37')
COUNT=module(ROOT/'review/understand_m43_t75/count.py','prior_r3_count')
# Charge unary negation omitted by original R3 type, and guard integer-power semantics.
COUNT.C['neg']=0
COUNT.Num.__neg__=lambda self:(COUNT.C.__setitem__('neg',COUNT.C['neg']+1) or COUNT.Num(-self.x))
def power(self,p):
 if not isinstance(p,int) or p<0:raise ValueError('uncounted general power')
 COUNT.C['mul']+=max(0,p-1);return COUNT.Num(self.x**p)
COUNT.Num.__pow__=power
FIB_PATH=ROOT/'review/understand_ladder/fibre_strict.py'
src=FIB_PATH.read_text();anchor='    stc = State(model, x, P, U, nn, ops)\n';assert src.count(anchor)==1
src=src.replace(anchor,anchor+'    cache.setdefault("_alive", []).append(stc)\n')
FIB=types.ModuleType('review_fibre');FIB.__file__=str(FIB_PATH);exec(compile(src,str(FIB_PATH)+' [review R4 cache fix]','exec'),FIB.__dict__);FIB.LOOK=3
from m43 import Ops
import exact as author_reference

def load(k,s):
 import torch
 path=Path('/workspace/work/t114_checkpoints')/f'2nd_argmax_4_3_{2**k}_{s}.pth'
 sd=torch.load(path,weights_only=True,map_location='cpu')
 return tuple(sd[n].double().numpy().copy() for n in('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))[0][:,0],sd['rnn.weight_hh_l0'].double().numpy().copy(),sd['linear.weight'].double().numpy().copy()
def rms(m):return float(np.sqrt(sum(float(np.sum(a*a)) for a in m)/sum(a.size for a in m)))
def noisy(m,ns,n,sigma):
 rng=np.random.default_rng(ns)
 return [tuple(a+sigma*rng.standard_normal(a.shape) for a in m) for _ in range(n)]
def t75(m):
 o=Ops();E75.STATS.update(cut=0,resolved=0,cells=0);v=E75.est75(m,o,**E75.RECIPE)
 own,c,proxy=COUNT.counted_recipe(m);sample=COUNT.counted_sample(m,[.3,-1.2,.7]);sample['correctness_comparison']=1
 return dict(value=v,claimed_ops=o.n,own_value=own,own_counts=c,own_ops=sum(c.values()),sample_counts=sample,sample_ops=sum(sample.values()),reimplementation_proxy=proxy)
def fibre(m):
 o=FIB.Ops();info={};v=FIB.est_fibre2(m,o,info)
 return dict(value=v,strict_ops=o.n+getattr(o,'x',0),traces=info.get('traces'))
