"""Pin installed rank/SVD wrappers, binaries, dispatch and observed call shapes."""
import ctypes,hashlib,inspect,json,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
LIB=Path(np.__file__).parent.parent/'numpy.libs/libscipy_openblas64_-f48b354e.so'
EXT=Path(np.linalg._umath_linalg.__file__)
EXPECTED={str(EXT):'2e075f582c3a25a21fe97a55931a0fdb22ceb84f4fce8203a5040d405ccf65ce',str(LIB):'432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time()
for p,sha in EXPECTED.items():assert digest(Path(p))==sha,p
for name in ('matrix_rank','svd'):(HERE/f'installed_{name}.py.txt').write_text(inspect.getsource(getattr(np.linalg,name).__wrapped__))
lib=ctypes.CDLL(str(LIB));versions=[]
for name in ('scipy_ilaver_64_','scipy_ilaver_'):
    try:f=getattr(lib,name)
    except AttributeError:continue
    integers=[ctypes.c_int64() for _ in range(3)];f(*[ctypes.byref(x) for x in integers]);versions.append(dict(symbol=name,version=[x.value for x in integers]))
symbols=subprocess.check_output(['nm','-C',str(EXT)],text=True)
selected=[line for line in symbols.splitlines() if ('svd<' in line or 'svd_N' in line or 'dgesdd' in line)]
(HERE/'numpy_svd_symbols.txt').write_text('\n'.join(selected)+'\n')
for symbol in ('scipy_dgesdd_64_','scipy_dbdsdc_64_','scipy_dlasq1_64_','scipy_dlasq2_64_'):
    result=subprocess.check_output(['objdump','-d','--disassemble='+symbol,str(LIB)],text=True);(HERE/(symbol+'_disassembly.txt')).write_text(result)
census={};pins={}
for name,folder in [('exact','plackett_cost_review'),('recipe','recipe_cost_review')]:
    p=HERE.parent/folder/'summary.json';pins[str(p)]=digest(p);saved=json.loads(p.read_text());shapes=Counter();options=Counter()
    for item in saved['opaque_calls']:
        if item['key'][1]=='np.linalg.matrix_rank':shapes[item['key'][2]]+=item['count'];options[(item['key'][2],item['key'][4])]+=item['count']
    census[name]=dict(count=sum(shapes.values()),shapes=dict(shapes),shape_options=[dict(shape=k[0],options=k[1],count=v) for k,v in sorted(options.items())])
result=dict(numpy=np.__version__,binary_hashes=EXPECTED,lapack_version_symbols=versions,census=census,source_ledger_hashes=pins,numpy_symbols=selected,cpu_seconds=time.process_time()-start)
(HERE/'environment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('census','numpy_symbols')}))
