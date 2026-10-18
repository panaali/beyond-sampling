"""Saved cold child/driver evidence and bounded numeric-body contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native/spectral')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
K=P/'svd_kernels';a=read(K/'capture.json');summary=read(K/'summary.json')
assert a['complete'] and a['runtime_table_guard'] and a['dispatch_restored'] and len(a['rows'])==6
assert a['pins_sha256']==sha(K/'pins.json') and a['runner_sha256']==sha(K/'capture.py') and a['spy_binary_sha256']==sha(K/'spy.so')
for name in ('pins.json','tiny_price_pins.json'):
    for p,h in read(K/name).items():assert sha(Path(p))==h
original=read(P.parent/'capture.json');counts=Counter();parts=Counter();sizes=[]
for r in a['rows']:
    orig=original['rows'][r['saved_row']]
    assert orig['name']=='svd' and r['parity'] and r['output_hex']==[x['hex'] for x in orig['output']['items']]
    e=r['events'];counts.update(x['kernel'] for x in e)
    gs=[x for x in e if x['kernel']=='gemv_n'];assert len(gs)==1
    g=gs[0];m=g['m'];assert g['n']==m and 1<=m<=3 and (g['lda'],g['ldb'],g['ldc'])==(4,4,1)
    sizes.append(m);parts['gemv']+=2*m*m+2*m
    xs=[x for x in e if x['kernel']=='axpy'];assert len(xs)==m+2
    for x in xs:assert x['m']==m and x['lda']==x['ldb']==1
    parts['axpy']+=sum(2*x['m'] for x in xs)
    parts['ger']+=m
    ss=[x for x in e if x['kernel']=='scal'];assert len(ss)==3
    assert [(x['m'],x['lda'],x['ldc']) for x in ss]==[(3,1,1),(m,1,0),(3,4,1)]
    assert ss[1]['alpha_bits']==0;parts['new_scale']+=m+3
    gg=[x for x in e if x['kernel']=='small_b0_nn'];assert len(gg)==1
    assert (gg[0]['m'],gg[0]['n'],gg[0]['k'])==(1,4,1)
    parts['small_b0_nn']+=4*(1*8*2+8)
assert sizes==[1,2,3,2,3,3] and dict(counts)==summary['kernel_calls']
price=read(K/'tiny_price_contraction.json');assert dict(parts)==price['parts'] and sum(parts.values())==price['sum']==850
# Separate additional driver attribution; old events must remain an exact subsequence.
D=P/'driver_census';d=read(D/'capture.json');old=read(P/'children/capture.json');s=read(D/'summary.json')
assert d['complete'] and len(d['rows'])==len(old['rows'])==12
assert d['manifest_sha256']==sha(D/'capture_manifest.json') and d['runner_sha256']==sha(D/'capture.py') and d['spy_binary_sha256']==sha(D/'spy.so')
for p,h in read(D/'capture_manifest.json')['files'].items():assert sha(Path(p))==h
oldnames={e['function'] for r in old['rows'] for e in r['events']};extra=Counter();zero_norm=0;ordinary=0
for x,y in zip(old['rows'],d['rows']):
    assert x['saved_row']==y['saved_row'] and x['name']==y['name']
    assert y['parity'] and y['output_hex']==x['output_hex']
    assert [e for e in y['events'] if e['function'] in oldnames]==x['events']
    extra.update(e['function'] for e in y['events'] if e['function'] not in oldnames)
    assert not any(e['function']=='dlascl' for e in y['events'])
    fg=[e for e in y['events'] if e['function']=='dlarfg' and e['args'][0]>1]
    nr=[e for e in y['events'] if e['function']=='dnrm2'];assert len(fg)==len(nr)
    if y['name']=='svd':
        assert len(fg)==1 and fg[0]['args'][0]==4 and fg[0]['args'][2]==0;ordinary+=1
        assert [e['args'] for e in y['events'] if e['function']=='dlange']==[[1,4,ord('M'),0]]
        assert all(e['args'][0]==1 for e in y['events'] if e['function']=='dbdsdc')
    else:
        assert all(e['args'][2]==1 for e in fg);zero_norm+=len(fg)
assert dict(extra)==s['extra_calls']
assert ordinary==6 and zero_norm==32
for p,h in read(D/'reflector_price_pins.json').items():assert sha(Path(p))==h
out=dict(complete=True,svd_kernel_calls=dict(counts),tiny_body_parts=dict(parts),tiny_body_plus_scale=850,
         extra_driver_calls=dict(extra),ordinary_reflectors=ordinary,zero_norm_reflectors=zero_norm,
         reflector_exclusive_native_categories=ordinary*19+zero_norm,
         kernel_capture_sha256=sha(K/'capture.json'),driver_capture_sha256=sha(D/'capture.json'),
         new_native_calls=0,
         scope='Fixed saved dispatch/body evidence only. Tiny body 850 includes new internal beta scaling, not old public SCAL already charged in helper 2424. Reflector 146 uses observed no-rescale and fixed binding. Remaining driver prices, wrapper/control/memory and native reduction gaps are separate.')
(HERE/'cold_kernel_driver_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
