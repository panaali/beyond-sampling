"""Recover only canonical public checkpoint bytes; never load or run a model."""
from pathlib import Path
import base64,hashlib,json,time,urllib.request
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=P/'checkpoint_identity.json';assert not dest.exists();t=time.process_time()
    p=json.loads((P/'protocol.json').read_text())
    out=dict(complete=False,claim=p['claim'],protocol_sha256=sha(P/'protocol.json'),source_sha256=sha(__file__),checkpoints=[])
    def save():out['cpu']=time.process_time()-t;dest.write_text(json.dumps(out,indent=2)+'\n')
    save()
    try:
        for seed,url in zip(p['seeds'],p['canonical_urls']):
            path=P/'checkpoints'/f'seed{seed}.pth';assert not path.exists()
            row=dict(seed=seed,canonical_url=url,local_path=str(path),status='started');out['checkpoints'].append(row);save()
            with urllib.request.urlopen(url,timeout=30) as resp:
                data=resp.read(p['maximum_checkpoint_bytes']+1)
                headers={k:resp.headers.get_all(k) for k in ('last-modified','etag','x-goog-generation','x-goog-hash','content-length')}
                row.update(headers=headers,http_status=resp.status,bytes=len(data));save()
            assert 0<len(data)<=p['maximum_checkpoint_bytes']
            assert int(headers['content-length'][0])==len(data)
            md5=base64.b64encode(hashlib.md5(data).digest()).decode()
            hash_members=[x.strip() for line in headers['x-goog-hash'] for x in line.split(',')]
            assert 'md5='+md5 in hash_members
            assert len(headers['x-goog-generation'])==1 and headers['x-goog-generation'][0].isdigit()
            path.write_bytes(data)
            row.update(status='complete',sha256=hashlib.sha256(data).hexdigest(),md5_base64=md5);save()
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps(dict(complete=out['complete'],checkpoints=len(out['checkpoints']),cpu=out['cpu'])))
if __name__=='__main__':main()
