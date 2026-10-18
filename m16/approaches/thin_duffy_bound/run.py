"""Bounded resumable exact-range/bound checks on frozen thin indices only."""
import time
START=time.process_time()
from pathlib import Path
import argparse,gzip,hashlib,json,resource,math
import numpy as np
import primitive as A
P=Path(__file__).resolve().parent
PROTOCOL=A.PROTOCOL

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()))
def write(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(gzip.compress(json.dumps(d,separators=(',',':'),allow_nan=False).encode(),mtime=0));q.replace(p)
def main(first,stop):
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));assert json.loads((P/'controls.json').read_text())['complete'];tracker,radial=A.load()
 for f,h in PROTOCOL['source_records'].items():assert sha(Path(f))==h
 for f,h in PROTOCOL['baseline_shards'].items():assert sha(Path(f))==h
 assert sha(P/'prior_certificates.json')==PROTOCOL['prior_certificates_sha256'];prior=json.loads((P/'prior_certificates.json').read_text());rays=np.load(PROTOCOL['rays'])['rays'];assert len(rays)==PROTOCOL['N']
 refs={r['index']:r for f in PROTOCOL['source_records'] for r in json.loads(Path(f).read_text())['rows']};assert sorted(refs)==list(range(PROTOCOL['N']))
 selected=PROTOCOL['indices'];assert selected==sorted(i for i,r in refs.items() if r['diagnostics']['branch']=='duffy10');assert 0<=first<stop<=math.ceil(len(selected)/256)
 session=dict(first=first,stop=stop,completed=[],complete=False,source_sha256=sha(Path(__file__)),primitive_sha256=sha(P/'primitive.py'),protocol_sha256=sha(P/'protocol.json'));sp=P/f'session_{first:03d}_{stop:03d}_{time.time_ns()}.json'
 def save():
  session.update(cpu=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);sp.write_text(json.dumps(session,indent=2)+'\n')
 for chunk in range(first,stop):
  ids=selected[256*chunk:256*(chunk+1)];path=P/f'shard_{chunk:03d}.json.gz';d=read(path) if path.exists() else dict(indices=ids,rows=[],complete=False,protocol_sha256=session['protocol_sha256'])
  assert d['indices']==ids and [r['index'] for r in d['rows']]==ids[:len(d['rows'])] and d['protocol_sha256']==session['protocol_sha256']
  if any(r['mode']=='failure' for r in d['rows']):session['stop_reason']='Preserved prior failure';save();return
  if d['complete']:session['completed'].append(chunk);continue
  for i in ids[len(d['rows']):]:
   if time.process_time()-START>18:write(path,d);session['stop_reason']='18CPU soft cap';save();return
   try:
    r=A.certificate(rays[i],tracker,radial,prior.get(str(i)));r.update(index=i,source_value=refs[i]['value'],reference=refs[i]['reference'],reference_bound=refs[i]['reference_bound'],observed_difference=refs[i]['difference'])
    if r['mode']=='bounded':
     r['quadrature']=A.bound(r)
     if i in PROTOCOL['precision_check_indices']:r['precision100']=A.bound(r,100)
   except Exception as exc:r=dict(index=i,mode='failure',reason=repr(exc))
   d['rows'].append(r)
   if r['mode']=='failure':write(path,d);session['stop_reason']='Explicit failure';save();return
  d['complete']=True;write(path,d);session['completed'].append(chunk);save()
 session['complete']=True;save();print(json.dumps(session))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('first',type=int);ap.add_argument('stop',type=int);a=ap.parse_args();main(a.first,a.stop)
