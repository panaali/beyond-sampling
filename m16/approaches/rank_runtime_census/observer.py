"""Read-only native-census interface with pinned installed binaries."""
from pathlib import Path
import ctypes,hashlib,json
HERE=Path(__file__).resolve().parent
ENV=json.loads((HERE.parent/'rank_cost_review/environment.json').read_text())
for filename,sha in ENV['binary_hashes'].items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==sha
LIB=ctypes.CDLL(str(HERE/'spy.so'))
LIB.census_size.restype=LIB.census_overflow.restype=ctypes.c_int64
NAMES=json.loads((HERE/'spy_manifest.json').read_text())['kernel_ids']
def reset():LIB.census_reset()
def report():
 assert LIB.census_overflow()==0
 n=LIB.census_size();assert 0<=n<=1024
 data=(ctypes.c_int64*(n*6))();LIB.census_get(data)
 rows=[]
 for i in range(n):
  kernel,m,ncol,query,count,fail=list(map(int,data[6*i:6*i+6]));rows.append(dict(kernel=NAMES[str(kernel)],m=m,n=ncol,query=bool(query),calls=count,failures=fail))
 native=(ctypes.c_int64*12)();LIB.nested_stats(native)
 gesdd=(ctypes.c_int64*12)();LIB.audit_stats(gesdd)
 return dict(kernels=sorted(rows,key=lambda x:(x['kernel'],x['m'],x['n'],x['query'])),nested=list(map(int,native)),gesdd=list(map(int,gesdd)),overflow=0)
