"""Repeat the same complete checks with only the radial adapter replaced."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
source=(HERE/'check.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='0c01fb970c14553e5bac24aa1992dd424cf0abe24aa4af8ebf5329b2ad3130ad'
old='from selector import Guarded,scalar_bound,scalar_radial,ORDERS'
new='from selector import scalar_bound,scalar_radial,ORDERS\nfrom pow_selector import PowGuarded as Guarded'
assert source.count(old)==1 and source.count("/'controls.json'")==1
source=source.replace(old,new).replace("/'controls.json'","/'pow_controls.json'")
exec(compile(source,str(HERE/'check.py')+' [scalar-pow rescue]','exec'),{'__file__':str(HERE/'check.py'),'__name__':'pow_controls'})
