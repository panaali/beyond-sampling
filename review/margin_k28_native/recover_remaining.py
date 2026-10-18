"""Preserve the first failure; read repeated checksum fields and reuse saved bytes."""
import base64,hashlib,json,time,traceback,urllib.request
from pathlib import Path
from recover_checkpoints import HERE,FIELDS
def headers(response):return {k:', '.join(response.headers.get_all(k)) for k in FIELDS if k in response.headers}
def main():
 target=HERE/'checkpoint_recovery_complete.json';assert not target.exists();priorpath=HERE/'checkpoint_recovery.json';prior=json.loads(priorpath.read_text());assert not prior['complete'] and len(prior['records'])==1 and prior['records'][0]['seed']==0
 inventory=json.loads((HERE/'inventory.json').read_text());assert inventory['complete']
 for p,d in inventory['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d
 out=dict(complete=False,claim='7372744e',prior_failure_sha256=hashlib.sha256(priorpath.read_bytes()).hexdigest(),inventory_sha256=hashlib.sha256((HERE/'inventory.json').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=[],historical_checkpoint_identity_verified=False,scope='Current canonical objects only. Initial single-header parser failure retained; seed0 payload reused, metadata fetched at same generation. No evaluator/reference/model-forward calls; inherited proxy/TLS unchanged.')
 start=time.monotonic()
 def save():
  out['wall_seconds']=time.monotonic()-start;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(target)
 try:
  for seed in range(5):
   url=inventory['seed_records'][seed]['canonical_checkpoint_url'];assert url==f'https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_268435456_{seed}.pth'
   p=HERE/'checkpoints'/f'k28_seed{seed}.pth'
   if seed==0:
    old=prior['records'][0];assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==old['sha256'];fetch=old['fetch_url'];generation=old['head_headers']['x-goog-generation']
    assert fetch==url+'?generation='+generation
    with urllib.request.urlopen(urllib.request.Request(fetch,method='HEAD'),timeout=20) as response:h=headers(response);assert response.status==200
    data=p.read_bytes();g=None
   else:
    assert not p.exists()
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=20) as response:h=headers(response);assert response.status==200
    generation=h['x-goog-generation'];assert generation.isdigit();fetch=url+'?generation='+generation
    with urllib.request.urlopen(fetch,timeout=20) as response:g=headers(response);assert response.status==200;data=response.read(1048577)
    p.write_bytes(data)
   r=dict(seed=seed,url=url,fetch_url=fetch,path=str(p),head_headers=h,get_headers=g,payload_reused=seed==0,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),md5_base64=base64.b64encode(hashlib.md5(data).digest()).decode(),validated=False)
   out['records'].append(r);save()
   assert 0<len(data)<=1048576
   for head in ([h] if g is None else [h,g]):
    assert head['x-goog-generation']==generation and int(head['content-length'])==len(data)
    assert 'md5='+r['md5_base64'] in [x.strip() for x in head['x-goog-hash'].split(',')]
   r['validated']=True;save()
  out['complete']=True;save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(complete=True,bytes=sum(r['bytes'] for r in out['records']),reused_payloads=sum(r['payload_reused'] for r in out['records']))))
if __name__=='__main__':main()
