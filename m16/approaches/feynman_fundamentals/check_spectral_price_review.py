"""Compose only reviewed named spectral subsets from saved calls."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native/spectral')
a=json.loads((P/'children/capture.json').read_text());assert a['complete']
price=Counter();orders=[];children=Counter()
for r in a['rows']:
    assert r['parity']
    for e in r['events']:
        k=e['function'];args=e['args'];children[k]+=1
        if k=='dlarfg':continue
        if k=='dnrm2':
            n,inc=args[:2];assert n>=1 and inc>0
            price[k]+=1 if n==1 else 2*n+4
        elif k=='dscal':
            n,inc=args[:2];assert n==3 and inc in (1,4);price[k]+=n+3
        elif k=='dlapy2':price[k]+=13
        elif k=='dlae2':price[k]+=28
        elif k=='dlasrt':
            n=args[0];assert 1<=n<=16;price[k]+=n*(n-1)//2
        elif k=='dlanst':
            n,norm=args[:2];assert norm==ord('M');price[k]+=8*n-6
        elif k=='dsterf':orders.append(args[0])
assert children['dlascl']==0 and orders==[6,3,3,6,10,16]
assert price==dict(dnrm2=484,dscal=72,dlapy2=1183,dlae2=168,dlasrt=201,dlanst=316)
source=sum(559*n*(n-1)+368*n+6 for n in orders)
installed=sum(626*n*(n-1)+520*n+6 for n in orders)
assert source==240946 and installed==274568
out=dict(complete=True,helper_numeric_subsets=dict(price),helper_sum=sum(price.values()),
         DSTERF_exclusive_source=source,DSTERF_exclusive_installed_category_upper=installed,
         source_and_installed_are_alternatives=True,
         pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'DSTERF_NATIVE.md',P/'dsterf_disassembly.txt',P/'children/HELPER_PRICES.md',P/'children/capture.json')},
         scope='Finite no-scale spectral trace. Packed abs/SIGN lanes, recurrence and cap correspondence reviewed from pinned disassembly. Parent/child categories are separate; other SVD/BLAS/reduction/utility/runtime costs remain open. No complete native bound.',new_native_calls=0)
(HERE/'spectral_price_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
