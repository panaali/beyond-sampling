"""Saved fixed-path driver composition, with native/source layers explicit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,re
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native')
D=P/'spectral/driver_census'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'capture.json');price=read(D/'driver_price_contraction.json');comp=read(P/'composition_candidate.json')
for name in ('driver_price_pins.json','utility_pins.json'):
    for p,h in read(D/name).items():assert sha(Path(p))==h
for p,h in comp['pins'].items():assert sha(Path(p))==h
events=[e for r in a['rows'] for e in r['events']];c=Counter(e['function'] for e in events)
numeric=re.compile(r'\b(?:addsd|subsd|mulsd|divsd|sqrtsd|ucomisd|comisd)\b')
for name,n in [('dgesdd',10),('dsyevd',9)]:
    lines=(D/f'scipy_{name}_64_.disasm.txt').read_text().splitlines()
    assert sum(bool(numeric.search(s)) for s in lines)==n
    assert c[name]==12
for name in ('dgelqf','dorglq','dgebrd','dsytrd','dormbr','dormqr','dorm2r'):
    assert not numeric.search((D/f'scipy_{name}_64_.disasm.txt').read_text())
assert all(e['args'][0]==130 for e in events if e['function']=='droundup_lwork')
assert not c['dlascl']
norms=[e['args'][0] for e in events if e['function']=='dlansy']
assert norms==[6,3,3,6,10,16]
assert all(e['args'][1:3]==[ord('M'),ord('L')] for e in events if e['function']=='dlansy')
parts=dict(dgesdd=c['dgesdd']*10,dgesdd_disnan=6,dsyevd=c['dsyevd']*9,
           dorgl2=c['dorgl2']*3,dsytd2=sum(n-1 for n in norms),dbdsdc=c['dbdsdc']*8,
           droundup=c['droundup_lwork'],dlange=c['dlange']*4*4,
           dlansy=4*sum(n*(n+1)//2 for n in norms),dlarf1f=6*(2*4+4+4*3+2)+6*(1+4+1+2),
           dlarfg=146,public_blas_permit=c['dgemv']*2+c['dger']+c['daxpy']*4+c['dgemm']*7+6*3,
           fixed_utilities=6)
assert parts==price['parts'] and sum(parts.values())==price['named_subset']==1914
assert comp['parts']==dict(inverse_48=7968,legendre_visible_numeric=6910,legendre_visible_integer=1018,
                          dsterf_exclusive_native=274568,spectral_helpers=2424,new_tiny_blas_bodies=850,
                          drivers_and_fixed_utilities=1914)
assert sum(comp['parts'].values())==comp['cold_named_subset']==295652
assert comp['prior_visible_upper']==1838399658
assert comp['conditional_composition']==comp['prior_visible_upper']+comp['cold_named_subset']==1838695310
out=dict(complete=True,driver_parts=parts,driver_named_categories=1914,cold_named_categories=295652,
         conditional_hybrid_composition=1838695310,composition_sha256=sha(P/'composition_candidate.json'),
         new_native_model_calls=0,
         scope='Fixed-path numerical-category composition only. Static driver sites require the recorded branch/shape premises; sources, selected native bodies and integer categories remain mixed. No double charge of old SCAL/new beta scaling or GER/AXPY children. Not a full native or hardware bound; ufunc/reduction internals, conversions, control/memory/runtime remain excluded.')
(HERE/'cold_composition_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
