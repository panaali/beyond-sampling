"""Fixed stored ranges, outward bound only; resumable256-index shards."""
import time
START=time.process_time()
from pathlib import Path
import argparse,gzip,hashlib,json,resource,math
import primitive as A
P=Path(__file__).resolve().parent
PROTOCOL=A.PROTOCOL

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()))
def write(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(gzip.compress(json.dumps(d,separators=(',',':'),allow_nan=False).encode(),mtime=0));q.replace(p)
def main(first,stop):
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));assert json.loads((P/'controls.json').read_text())['complete']
 for f,h in PROTOCOL['pins'].items():assert sha(Path(f))==h
 rows={r['index']:r for p in sorted(P.parent.glob('shard_*.json.gz')) for r in read(p)['rows']};rescues=json.loads((P.parent/'rescue/results.json').read_text());assert rescues['complete'];replacement={r['index']:r for r in rescues['rows']};assert len(replacement)==54
 ids=sorted(rows);assert ids==PROTOCOL['indices'];bound=A.Bound();assert 0<=first<stop<=math.ceil(len(ids)/256)
 session=dict(first=first,stop=stop,completed=[],complete=False,cold_counts=bound.cold_counts,source_sha256=sha(Path(__file__)),primitive_sha256=sha(P/'primitive.py'),protocol_sha256=sha(P/'protocol.json'));sp=P/f'session_{first:03d}_{stop:03d}_{time.time_ns()}.json'
 def save():session.update(cpu=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);sp.write_text(json.dumps(session,indent=2)+'\n')
 for chunk in range(first,stop):
  selected=ids[256*chunk:256*(chunk+1)];path=P/f'shard_{chunk:03d}.json.gz';d=read(path) if path.exists() else dict(indices=selected,rows=[],complete=False,protocol_sha256=session['protocol_sha256'])
  assert d['indices']==selected and [r['index'] for r in d['rows']]==selected[:len(d['rows'])] and d['protocol_sha256']==session['protocol_sha256']
  if any(r.get('failure') for r in d['rows']):session['stop_reason']='Preserved failure';save();return
  if d['complete']:session['completed'].append(chunk);continue
  for i in selected[len(d['rows']):]:
   if time.process_time()-START>18:write(path,d);session['stop_reason']='18CPU soft cap';save();return
   source=replacement[i]['recheck'] if i in replacement else rows[i]
   try:r=bound.one(source);r.update(index=i,reused_kkt_rescue=i in replacement)
   except Exception as exc:r=dict(index=i,failure=repr(exc))
   d['rows'].append(r)
   if r.get('failure'):write(path,d);session['stop_reason']='Explicit failure';save();return
  d['complete']=True;write(path,d);session['completed'].append(chunk);save()
 session['complete']=True;save();print(json.dumps(session))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('first',type=int);ap.add_argument('stop',type=int);a=ap.parse_args();main(a.first,a.stop)
