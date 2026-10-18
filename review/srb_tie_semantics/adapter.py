"""Extract only unchanged h9/Q, then replace exactly the tie predicate."""
from pathlib import Path
import ast,hashlib
H=Path(__file__).resolve().parent
OLD='never = (dB == 0) & (dA <= 0)'
NEW='never = (dB == 0) & ((dA < 0) | ((dA == 0) & (np.arange(A.shape[1])[None, :] < tgt[:, None])))'
SOURCE_SHA='487d864f7389c7539b6b5acad06853a3b25e4c2c6a76b49d15817a87ba734958'
def load():
    source=(H/'rb9_snapshot.py').read_text();assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_SHA
    tree=ast.parse(source);functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    assert [n.name for n in functions]==['h9','Q']
    text='\n\n'.join(ast.get_source_segment(source,n) for n in functions)
    assert text.count(OLD)==1
    patched=text.replace(OLD,NEW)
    # Reversal proves no incidental source change.
    assert ast.dump(ast.parse(patched.replace(NEW,OLD)))==ast.dump(ast.parse(text))
    import numpy as np
    from scipy.special import ndtr,stdtr
    env=dict(np=np,ndtr=ndtr,stdtr=stdtr,T=10); old=env.copy();new=env.copy()
    exec(compile(text,'<original rb9 functions>','exec'),old)
    exec(compile(patched,'<local tie repair>','exec'),new)
    return old['Q'],new['Q'],dict(original_extracted_sha256=hashlib.sha256(text.encode()).hexdigest(),patched_sha256=hashlib.sha256(patched.encode()).hexdigest())
