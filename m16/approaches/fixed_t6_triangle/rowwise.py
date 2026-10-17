"""Narrow T104 projection rescue; original primitive remains immutable."""
import hashlib
from pathlib import Path
import primitive
source=Path(primitive.__file__).read_text()
SOURCE_SHA=hashlib.sha256(source.encode()).hexdigest()
if SOURCE_SHA!='6b4f511eab0309f4c296ef374124c429fc2b421001dc01b8ebea27c3d8b3e1e7':
    raise ValueError('fixed-frame source changed; requires fresh comparison')
old='    a1=A@e1; a2=A@e2'
assert source.count(old)==1
source=source.replace(old,"    a1=np.einsum('ij,j->i',A,e1,optimize=False); a2=np.einsum('ij,j->i',A,e2,optimize=False)")
namespace={}
exec(compile(source,'<T104 rowwise projection derivative>','exec'),namespace)
prepare=namespace['prepare'];evaluate=namespace['evaluate']
