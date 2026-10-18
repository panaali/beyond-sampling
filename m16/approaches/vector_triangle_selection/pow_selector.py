"""One-expression radial rescue; initial selector source remains unchanged."""
from pathlib import Path
import hashlib,types
from attribute import hybrid_radial

HERE=Path(__file__).resolve().parent
source=(HERE/'selector.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='392af5d931669dcd48d51cd72b75beab43f6ba9e4745cfc2efad45c61d50cf19'
old='radial=np.array([scalar_radial(v) for v in vertices]);mid=radial[:,2];ratio=radial[:,3]'
new='radial=hybrid_radial(vertices);mid=radial[:,2];ratio=radial[:,3]\n        self.audit.add("hybrid_radial_batch_and_scalar_pow",160*len(vertices))'
assert source.count(old)==1
module=types.ModuleType('preserved_guarded_selector_with_scalar_pow');module.__file__=str(HERE/'selector.py')
module.__dict__['hybrid_radial']=hybrid_radial
exec(compile(source.replace(old,new),str(HERE/'selector.py')+' [scalar-pow radial rescue]','exec'),module.__dict__)
PowGuarded=module.Guarded
