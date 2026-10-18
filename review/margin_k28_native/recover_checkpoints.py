"""Retrieve only the five canonical objects fixed by the original k28 generator."""
import base64,hashlib,json,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
FIELDS=('content-length','last-modified','etag','x-goog-generation','x-goog-metageneration','x-goog-hash','x-goog-stored-content-length')
def headers(response):return {k:response.headers[k] for k in FIELDS if k in response.headers}
def main():
 target=HERE/'checkpoint_recovery.json';assert not target.exists();inventory=json.loads((HERE/'inventory.json').read_text());assert inventory['complete'] and not inventory['model_identity_verified']
 for p,d in inventory['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d
 folder=HERE/'checkpoints';folder.mkdir(exist_ok=True)
 out=dict(complete=False,claim='7372744e',inventory_sha256=hashlib.sha256((HERE/'inventory.json').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=[],historical_checkpoint_identity_verified=False,scope='Current canonical objects only; prior bytes/per-copy identity were not saved. No evaluator/reference/model-forward call. Inherited proxy and TLS verification unchanged.')
 start=time.monotonic()
 def save():
  out['wall_seconds']=time.monotonic()-start;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(target)
 try:
  for seed in range(5):
   url=inventory['seed_records'][seed]['canonical_checkpoint_url'];assert url==f'https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_268435456_{seed}.pth'
   p=folder/f'k28_seed{seed}.pth';assert not p.exists()
   with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=20) as response:h=headers(response);assert response.status==200
   generation=h['x-goog-generation'];assert generation.isdigit();fetch=url+'?generation='+generation
   with urllib.request.urlopen(fetch,timeout=20) as response:
    g=headers(response);assert response.status==200;data=response.read(1048577)
   p.write_bytes(data)
   r=dict(seed=seed,url=url,fetch_url=fetch,path=str(p),head_headers=h,get_headers=g,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),md5_base64=base64.b64encode(hashlib.md5(data).digest()).decode(),validated=False)
   out['records'].append(r);save()
   assert 0<len(data)<=1048576 and g['x-goog-generation']==generation
   assert int(h['content-length'])==len(data) and int(g['content-length'])==len(data)
   for head in (h,g):assert 'md5='+r['md5_base64'] in [x.strip() for x in head['x-goog-hash'].split(',')]
   r['validated']=True;save()
  out['complete']=True;save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(complete=True,bytes=sum(r['bytes'] for r in out['records']),generations=[r['head_headers']['x-goog-generation'] for r in out['records']])));
if __name__=='__main__':main()
