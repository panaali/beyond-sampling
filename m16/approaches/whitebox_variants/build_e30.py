import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SRC=Path('/tmp/whitebox-review-e30/estimator.py')
assert hashlib.sha256(SRC.read_bytes()).hexdigest()=='6ddd97a96c30397683815c530bfb422f0a49ffdea49e4f7827149ed0a14729b0'
src=SRC.read_text()
variants={'e30_rank384':{'V21_R_OLD':('416','384'),'V24_R_OLD2':('224','192')},'e30_rank448':{'V21_R_OLD':('416','448')},'e30_tau27':{'X_DEAD_TAU':('2.85','2.7')},'e30_tau30':{'X_DEAD_TAU':('2.85','3.0')}}
manifest=json.loads((HERE/'sources.json').read_text())
for name,changes in variants.items():
 s=src
 for key,(old,new) in changes.items():
  needle=f'_os.environ.get("{key}", "{old}")';assert needle in s;s=s.replace(needle,repr(new))
 s='# T476 derivative of MIT-licensed e30_a; see E30_LICENSE. Fixed changes: '+repr(changes)+'\n'+s
 p=HERE/f'estimator_{name}.py';p.write_text(s)
 manifest[name]={'local_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':changes}
(HERE/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'E30_LICENSE').write_bytes(SRC.with_name('LICENSE').read_bytes())
p=HERE/'run.py';s=p.read_text().replace('1_650_000_000','7_500_000_000').replace('RSS guard exceeded 1.65 GB','RSS guard exceeded 7.5 GB');p.write_text(s)
