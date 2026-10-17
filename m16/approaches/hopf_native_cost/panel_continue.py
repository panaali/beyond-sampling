"""Frozen remaining23 rows, same reviewed pilot body and output contract."""
import hashlib,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE/'panel_probe.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='793112e49c5fbb35066f517eba2a4b546ff9d525afe5482d3c78eaa16ae34797'
src=BASE.read_text()
old="assert (args.seed,args.copy)==(0,1),'Only frozen first-noise pilot authorized by this entrypoint'"
new="assert 0<=args.seed<5 and 0<=args.copy<5 and (args.seed,args.copy) not in ((0,0),(0,1)),'Only remaining frozen23 existing rows permitted'"
assert src.count(old)==1;src=src.replace(old,new)
old="'noise_snapshot.py','panel_probe.py']"
new="'noise_snapshot.py','panel_probe.py','panel_continue.py']"
assert src.count(old)==1;src=src.replace(old,new)
module=types.ModuleType('frozen_remaining_hopf_panel')
exec(compile(src,str(BASE)+' [remaining23 frozen only]','exec'),module.__dict__)
if __name__=='__main__':module.main()
