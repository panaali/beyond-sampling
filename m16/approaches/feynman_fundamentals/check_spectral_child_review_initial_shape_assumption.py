"""Saved child attribution only; no rule or spectral computation."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native/spectral')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=read(P/'capture.json');a=read(P/'children/capture.json');s=read(P/'children/summary.json')
assert a['complete'] and len(a['rows'])==12
assert a['manifest_sha256']==sha(P/'children/capture_manifest.json')
assert a['runner_sha256']==sha(P/'children/capture.py') and a['spy_binary_sha256']==sha(P/'children/spy.so')
for path,h in read(P/'children/capture_manifest.json')['files'].items():assert sha(Path(path))==h
original_names={e['function'] for r in old['rows'] for e in r['events']}
c=Counter(); shifts=[]; svd_shapes=[]
for previous,r in zip(old['rows'],a['rows']):
    assert r['saved_row']==previous['saved_row'] and r['name']==previous['name']
    assert r['parity'] and r['output_hex']==previous['output_hex']
    assert [e for e in r['events'] if e['function'] in original_names]==previous['events']
    c.update(e['function'] for e in r['events'])
    if r['name']=='leggauss':
        D=list(map(float.fromhex,r['sterf_input_D_hex']))
        E=list(map(float.fromhex,r['sterf_input_E_hex']))
        assert len(E)==len(D)-1 and all(x==0 for x in D)
        assert all(sys.float_info.min<=x<1 and math.isfinite(x) for x in E)
        assert [e['args'] for e in r['events'] if e['function']=='dsterf']==[[len(D),0,0,0]]
        shifts.append(sum(e['function']=='dlapy2' for e in r['events']))
    else:
        svd_shapes.append([e for e in r['events'] if e['function'] in ('dgemm','dgemv','daxpy','dger','dscal')])
assert dict(c)==s['function_calls'] and c['dlascl']==0
assert shifts==s['legendre_shift_calls']==[11,4,4,11,21,34]
assert s['all_DSTERF_D_zero'] and s['all_DSTERF_E_positive_normal'] and s['no_scale_calls']
assert all(x==svd_shapes[0] for x in svd_shapes)
out=dict(complete=True,outputs=12,original_event_subsequences_unchanged=True,
         child_counts=dict(c),legendre_shift_calls=shifts,svd_child_shapes=svd_shapes[0],
         capture_sha256=sha(P/'children/capture.json'),new_spectral_model_calls=0,
         scope='Finite saved child attribution with guarded current DSTERF input buffers. Zero observed scaling and fixed shapes do not price other routines or establish full native cost.')
(HERE/'spectral_child_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
