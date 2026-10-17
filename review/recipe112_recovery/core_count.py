"""Frozen existing64-model core shard, or existing32-model borderline shard."""
import time
START=time.process_time()
import argparse,hashlib,json,resource
from pathlib import Path
import counter_panel as P
HERE=P.HERE;R=P.R;M=P.M
def main(H,zs,noise,borderline=False):
 assert (borderline and (H,zs,noise)==(8,2,90)) or (not borderline and H in (4,5,6,8) and 0<=zs<5 and noise in (80,81,82,83))
 name='borderline' if borderline else f'core_H{H}_z{zs}_n{noise}';path=HERE/'direct_counts'/f'{name}.json';path.parent.mkdir(exist_ok=True);assert not path.exists(),'Preserve complete or partial prior run; do not overwrite'
 src=HERE/'snapshot'/'out'/('fresh_H8_rel0.0025_90.json' if borderline else f'fresh_H{H}_abs0.05_80-81-82-83.json');raw=json.loads(src.read_text());saved=raw['zoo'][str(zs)]['rows'][1:];n=raw['n'];assert n==(32 if borderline else 64);offset=raw['seeds'].index(noise)*n;saved=saved[offset:offset+n]
 base,weightpath=P.weights(H,zs);models=M.noisy_rel(base,n,.0025,noise) if borderline else M.noisy_abs(base,n,.05,noise);extra=[0]
 def eq(self,other):R.C['n']+=1;extra[0]+=1;return self.x==R.val(other)
 def ne(self,other):R.C['n']+=1;extra[0]+=1;return self.x!=R.val(other)
 R.Num.__eq__=eq;R.Num.__ne__=ne
 out=dict(H=H,zoo=zs,noise=noise,borderline=borderline,complete=False,rows=[],expected=n,checkpoint_sha256=hashlib.sha256(weightpath.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def save():
  out['cpu']=time.process_time()-START;out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;path.write_text(json.dumps(out,indent=2)+'\n')
 for i,(model,old) in enumerate(zip(models,saved)):
  row=dict(copy=i,truth=old['truth'],saved_estimate=old['est'],saved_ops=old['ops'],saved_cells=old['cells'],model_sha256=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest());out['rows'].append(row)
  try:
   stats={};R.C['n']=0;extra[0]=0;t=time.process_time();value=R.rec112(R.as_lists(model,True),R.P,stats);row.update(estimate=value,ops=R.C['n'],equality_ops=extra[0],stats=stats,kernel_cpu=time.process_time()-t,parity_difference=value-old['est'],parity_pass=abs(value-old['est'])<=1e-12,cell_count_equal=stats['cells']==old['cells'])
  except Exception as exc:row['failure']=repr(exc)
  if len(out['rows'])%8==0:save()
  if time.process_time()-START>50:out['budget_stop']=True;save();return
 out.update(complete=True,sampler_ops=R.sample_ops(H));out['all_parity_pass']=all(r.get('parity_pass',False) for r in out['rows']);out['max_difference']=max((abs(r.get('parity_difference',float('inf'))) for r in out['rows']),default=0);save();print(json.dumps(dict(shard=name,complete=out['complete'],parity=out['all_parity_pass'],max_difference=out['max_difference'],cpu=out['cpu'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('H',type=int);p.add_argument('zoo',type=int);p.add_argument('noise',type=int);p.add_argument('--borderline',action='store_true');a=p.parse_args();main(a.H,a.zoo,a.noise,a.borderline)
