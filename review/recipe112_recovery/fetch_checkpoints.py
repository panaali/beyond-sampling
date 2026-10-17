from pathlib import Path
import urllib.request,hashlib,json,time
HERE=Path(__file__).resolve().parent;DEST=HERE/'checkpoints';DEST.mkdir(exist_ok=True);rows=[];start=time.monotonic()
for H in (5,6):
 for seed in range(5):
  name=f'2nd_argmax_{H}_3_1073741824_{seed}.pth';path=DEST/name;url='https://storage.googleapis.com/arc-ml-public/alg/zoo/'+name
  if not path.exists():
   data=urllib.request.urlopen(url,timeout=30).read();assert 0<len(data)<1000000;path.write_bytes(data)
  rows.append(dict(name=name,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
(HERE/'checkpoint_manifest.json').write_text(json.dumps(dict(files=rows,wall_seconds=time.monotonic()-start),indent=2)+'\n');print(len(rows),'public checkpoint files pinned')
