"""Bounded resumable256-index shards; no completed cone or pilot is rerun."""
import time
START=time.process_time()
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,math,resource
import numpy as np
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())
PILOT=P.parent/'positive_h8_cost'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()))
def write(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(gzip.compress(json.dumps(d,allow_nan=False,separators=(',',':')).encode(),mtime=0));tmp.replace(p)
def main(first,stop):
 resource.setrlimit(resource.RLIMIT_CPU,(20,22))
 for f,h in PROTOCOL['pilot_pins'].items():assert sha(Path(f))==h,f
 spec=importlib.util.spec_from_file_location('frozen_h8_pilot',PILOT/'run.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 old,kernel,rescue,radial=M.load();rules=kernel.Rules();rays=np.load(M.REF/'error_captured.npz')['rays'];assert len(rays)==PROTOCOL['N'];pilot={r['index']:r for r in json.loads((PILOT/'pilot.json').read_text())['rows']};ref=np.zeros(len(rays));bd=np.zeros(len(rays));seen=np.zeros(len(rays),int)
 for k in range(8):
  a=np.load(M.REF/f'error_batch_{k}.npz');ref[a['indices']]=a['positive'];bd[a['indices']]=a['bounds'];seen[a['indices']]+=1
 assert np.all(seen==1);assert 0<=first<stop<=math.ceil(len(rays)/256)
 session=dict(first=first,stop=stop,complete_shards=[],source_sha256=sha(Path(__file__)),protocol_sha256=sha(P/'protocol.json'),complete=False)
 sessionpath=P/f'session_{first:03d}_{stop:03d}.json'
 def save_session():
  session.update(cpu=time.process_time()-START,cold_rule_events=rules.events,cold_rule_counts=rules.ledger.dict(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  sessionpath.write_text(json.dumps(session,indent=2)+'\n')
 for chunk in range(first,stop):
  path=P/f'shard_{chunk:03d}.json.gz';indices=list(range(256*chunk,min(256*(chunk+1),len(rays))))
  d=read(path) if path.exists() else dict(chunk=chunk,indices=indices,rows=[],complete=False,protocol_sha256=session['protocol_sha256'])
  assert d['indices']==indices and d['protocol_sha256']==session['protocol_sha256']
  assert [r['index'] for r in d['rows']]==indices[:len(d['rows'])]
  if d['complete']:session['complete_shards'].append(chunk);continue
  for index in indices[len(d['rows']):]:
   if time.process_time()-START>18:
    d['stop_reason']='18CPU soft cap';write(path,d);session['stop_reason']=d['stop_reason'];save_session();print(json.dumps(session));return
   if index in pilot:row=pilot[index].copy();row['reused_pilot']=True
   else:
    row=M.one(rays[index],old,kernel,rescue,radial,rules);row.update(index=index,reference=float(ref[index]),reference_bound=float(bd[index]),reused_pilot=False)
    if row['mode']=='returned':row.update(error=row['value']-row['reference'],comparison_pass=bool(abs(row['value']-row['reference'])<=row['bound']+row['reference_bound']+5e-14))
   d['rows'].append(row)
   if row['mode']=='failure' or row.get('comparison_pass') is False:
    d['stop_reason']='Explicit failure/comparison gate';write(path,d);session['stop_reason']=d['stop_reason'];save_session();print(json.dumps(session));return
  d.pop('stop_reason',None);d['complete']=True;write(path,d);session['complete_shards'].append(chunk);save_session()
 session['complete']=True;save_session();print(json.dumps({k:v for k,v in session.items() if k!='cold_rule_events'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('first',type=int);p.add_argument('stop',type=int);a=p.parse_args();main(a.first,a.stop)
