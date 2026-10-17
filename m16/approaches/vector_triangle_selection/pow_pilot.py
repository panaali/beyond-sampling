"""Same64-cone cold pilot with the isolated scalar-pow radial rescue."""
from pathlib import Path
import hashlib,types
HERE=Path(__file__).resolve().parent
source=(HERE/'pilot.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='fa3a1bcc42fa3f1ecdbd2ce8b8e056c1c423216dae357bf07aede240677c0e96'
old='from selector import Guarded';new='from pow_selector import PowGuarded as Guarded'
assert source.count(old)==1 and source.count("/'pilot.json'")==1
source=source.replace(old,new).replace("/'pilot.json'","/'pow_pilot.json'")
module=types.ModuleType('scalar_pow_pilot');module.__file__=str(HERE/'pilot.py')
exec(compile(source,str(HERE/'pilot.py')+' [scalar-pow rescue]','exec'),module.__dict__)
if __name__=='__main__':module.main()
