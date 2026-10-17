"""Complete original125 count cases, one frozen H per bounded process."""
import time
START=time.process_time()
import argparse,hashlib,json,resource
from pathlib import Path
import numpy as np
import semantics as S
import torch
HERE=S.HERE;R=S.R;M=S.M;E=S.E;Ops=S.Ops

def weights(H,zs):
 root=HERE/'checkpoints' if H in (5,6) else Path('/workspace/work/t114_checkpoints' if H==4 else '/workspace/work/t45_weights')
 path=root/f'2nd_argmax_{H}_3_1073741824_{zs}.pth';sd=torch.load(path,map_location='cpu',weights_only=True)
 return (sd['rnn.weight_ih_l0'].numpy().astype(float)[:,0],sd['rnn.weight_hh_l0'].numpy().astype(float),sd['linear.weight'].numpy().astype(float)),path

def main(H):
 assert H in (4,5,6,8,16);saved=json.loads((HERE/'snapshot'/'out'/'count.json').read_text())[str(H)];out=dict(H=H,rows=[],complete=False,checkpoint_hashes={},source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'counter_panel.py',HERE/'snapshot'/'rec112.py',HERE/'dependencies'/'understand_m43'/'est75.py')})
 extra=[0]
 def eq(self,other):R.C['n']+=1;extra[0]+=1;return self.x==R.val(other)
 def ne(self,other):R.C['n']+=1;extra[0]+=1;return self.x!=R.val(other)
 def save():
  out['cpu']=time.process_time()-START;out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;(HERE/f'counter_H{H}.json').write_text(json.dumps(out,indent=2)+'\n')
 for zs in range(5):
  base,path=weights(H,zs);out['checkpoint_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  for i,model in enumerate([base]+M.noisy_abs(base,4,.05,89)):
   old=saved['rows'][zs*5+i];assert old['zoo']==zs;row=dict(zoo=zs,copy=i-1,saved_original=old);out['rows'].append(row)
   try:
    op=Ops();t=time.process_time();value=E.est75(model,op,**E.RECIPE112);plain=R.rec112(R.as_lists(model));row.update(author=value,author_counter=op.n,float_reimplementation=plain,plain_cpu=time.process_time()-t);assert op.n==old['charge']
    R.Num.__eq__=eq;R.Num.__ne__=ne;extra[0]=0;t=time.process_time();fixed,count=R.counted(model);row.update(repaired=fixed,repaired_counter=count,equality_operations=extra[0],repaired_cpu=time.process_time()-t,parity_difference=fixed-plain,author_difference=fixed-value,parity_pass=abs(fixed-plain)<=1e-12)
   except Exception as exc:row['failure']=repr(exc)
   finally:S.equality_remove()
   save()
   if time.process_time()-START>50:out['budget_stop']=True;save();return
 out.update(complete=True,sampler_original=R.sample_ops(H));assert out['sampler_original']==saved['sample_auto'];valid=[r for r in out['rows'] if 'failure' not in r];out['all_parity_pass']=len(valid)==25 and all(r['parity_pass'] for r in valid);out['max_parity_difference']=max(abs(r['parity_difference']) for r in valid);out['max_author_difference']=max(abs(r['author_difference']) for r in valid);out['original_count_sum']=sum(r['saved_original']['auto'] for r in valid);out['repaired_count_sum']=sum(r['repaired_counter'] for r in valid);out['equality_count_sum']=sum(r['equality_operations'] for r in valid);save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','checkpoint_hashes','source_hashes')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('H',type=int);main(p.parse_args().H)
