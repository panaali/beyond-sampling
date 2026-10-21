import hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'whitebox_cumulants'
src=(BASE/'estimator_blocked.py').read_text()
variants={'block256':{},'rank320':{'V21_R_OLD':'320','V24_R_OLD2':'192'},'age3':{'V21_AGE_OLD':'3'},'corrected':{'V17_NO_CORR':'0'}}
manifest={}
for name,override in variants.items():
    s=re.sub(r'_os\.environ\.get\("([^"]+)", "([^"]*)"\)',lambda m:repr(override.get(m[1],m[2])),src)
    s=s.replace('min(128, n)','min(256, n)').replace('range(0, n, 128)','range(0, n, 256)').replace('lo + 128','lo + 256')
    s='# T476 fixed-knob variant '+name+'; derived from licensed T474 V25.\n'+s
    p=HERE/f'estimator_{name}.py';p.write_text(s)
    manifest[name]={'local_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':override,'block_size':256}
(HERE/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'LICENSE').write_bytes((BASE/'LICENSE').read_bytes())
s=(BASE/'run.py').read_text().replace('OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2"','OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"')
(HERE/'run.py').write_text(s)
