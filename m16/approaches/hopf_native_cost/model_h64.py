"""Only fixed grid/output/reference-key substitutions in the reviewed h8 runner."""
import hashlib,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE/'model_probe.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='15ffcf918dafaf665b6e0374579ac997731974df8824ec303e3f746235eb6568'
src=BASE.read_text()
changes=[
 ('model_probe_{args.arm}.json','model_h64_{args.arm}.json',1),
 ("'claim':'fec66d8b'","'claim':'c57b8cb6'",1),
 ("'nu':8,'nd':16","'nu':64,'nd':128",1),
 ("saved['h8']","saved['h64']",2),
 ("saved['h8_ops']","saved['h64_ops']",2),
 ('est_hopf(model,8,bo,nd=16)','est_hopf(model,64,bo,nd=128)',1),
 ('est_hopf(model,8,ops,nd=16)','est_hopf(model,64,ops,nd=128)',1),
 ("'model_identity.py','model_identity.json'","'model_identity.py','model_probe.py','model_h64.py','model_identity.json'",1),
 ('Single existing h8 model/recipe discriminator. No h64/panel transfer','Single existing h64 model/recipe discriminator. No panel transfer',1),
]
for old,new,n in changes:
 assert src.count(old)==n,(old,src.count(old),n)
 src=src.replace(old,new)
module=types.ModuleType('fixed_h64_runner');exec(compile(src,str(BASE)+' [fixed nu64 only]','exec'),module.__dict__)
if __name__=='__main__':module.main()
