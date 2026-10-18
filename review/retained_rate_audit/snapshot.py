"""Byte transport only; never import producer or decode retained numerical payloads."""
import hashlib,json,pathlib,subprocess
BASE=pathlib.Path(__file__).resolve().parent
records=[]
def git(*args): return subprocess.check_output(['git',*args])
def take(rev,path,dest):
    data=git('show',rev+':'+path); out=BASE/'frozen'/dest
    assert not out.exists(),out
    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    records.append(dict(revision=git('rev-parse',rev).decode().strip(),source=path,local=str(out.relative_to(BASE)),blob=git('rev-parse',rev+':'+path).decode().strip(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
for tag,rev,folder in [('t271','ac8669e0','rate_alpha'),('t277','e3033f5a','rate_T')]:
    root='m16/approaches/'+folder+'/'
    for path in git('ls-tree','-r','--name-only',rev,root).decode().splitlines(): take(rev,path,tag+'/'+path[len(root):])
for tag,rev,path in [('t271','ac8669e0','results/leaderboard_truths.json'),('t271','ac8669e0','m16/leaderboard.py'),('t271','ac8669e0','m16/approaches/sci_tao/estimator.py'),('t277','e3033f5a','m16/approaches/sci_tao/estimator.py')]: take(rev,path,tag+'/dependencies/'+path)
for h in (8,16):
    p=f'm16/approaches/width16_probe/anatomy/H{h}_T4_zoo0_chamber.json';take('ac8669e0',p,'t271/dependencies/'+p)
for p in git('ls-tree','-r','--name-only','e3033f5a','m16/approaches/width16_probe/res').decode().splitlines():
    if any(p.endswith(f'H16_T{t}_zoo{s}_exact.json') for t in (3,4,5,6,8,10) for s in range(3)):take('e3033f5a',p,'t277/dependencies/'+p)
for rev,folder in [('18f7e682','rate_alpha'),('d8d9b776','rate_T'),('80ea70ca','rate_T'),('a31f86ff','rate_T')]:take(rev,'m16/approaches/'+folder+'/README.md','history/'+rev+'_README.md')
(BASE/'manifest.json').write_text(json.dumps(dict(complete=True,operation='git byte transport only; numerical files not decoded',records=records),indent=2)+'\n')
print(len(records),'byte-pinned files')
