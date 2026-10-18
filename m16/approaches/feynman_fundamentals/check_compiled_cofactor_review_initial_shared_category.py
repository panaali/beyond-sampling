"""Saved compiled parity, exact interval and restored-environment audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,struct,math
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/cofactor3_guard');C=P/'compiled'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(x):return struct.pack('>d',x).hex()
def fl(x):return struct.unpack('>d',bytes.fromhex(x))[0]
a=read(C/'results.json');parent=read(P/'results.json');saved=read(P/'saved_minors/results.json')
assert a['complete'] and len(a['rows'])==89 and a['kernel_calls']==93
for path,h in a['source_pins'].items():assert sha(C/path)==h
for path,h in a['runtime_libc_libm'].items():assert sha(Path(path))==h
b=a['build'];assert sha(C/'guard.c')==b['source_sha256'] and sha(C/'build1/guard.so')==b['library_sha256']
assert sha(C/'build1/disassembly.txt')==b['disassembly_sha256']
for r,old in zip(a['rows'][:25],parent['rows']):
    assert r['name']==old['name'] and r['input_hex']==old['input_hex']
    q=old['result'];status={'fast_enclosure':0,'exact_fallback':1,'unknown_nonfinite':2}[q['status']]
    assert r['status']==status
    if status==0:
        assert r['bits']==[bits(float.fromhex(q[k])) for k in ('point','lower','upper','permanent','error_allowance')]
    elif status==1:
        assert r['outputs_untouched'] and r['fallback_bits']==[bits(float.fromhex(q[k])) for k in ('point','lower','upper')]
        # The original ledger includes the preceding Python guard; the new
        # fallback ledger contains only the explicitly invoked exact fallback.
        for key,value in r['fallback_counts'].items():assert q['counts'][key]==value
    else:assert r['outputs_untouched']
for r,old in zip(a['rows'][25:],saved['rows']):
    assert r['status']==0 and r['minor_hex']==old['minor_hex'] and r['cofactor_sign']==old['cofactor_sign']
    x=list(map(fl,r['bits']));point,lo,hi=x[:3]
    if r['cofactor_sign']<0:point,lo,hi=-point,-hi,-lo
    assert [bits(point),bits(lo),bits(hi)]==r['signed_bits']==[bits(float.fromhex(z)) for z in [old['candidate_signed_point'],*old['interval']]]
    exact=F(*map(int,old['exact_signed']));assert F(lo)<=exact<=F(hi)
    vals=[abs(float.fromhex(z)) for row in r['minor_hex'] for z in row]
    permanent=(vals[0]*(vals[4]*vals[8]+vals[5]*vals[7])+vals[1]*(vals[3]*vals[8]+vals[5]*vals[6]))+vals[2]*(vals[3]*vals[7]+vals[4]*vals[6])
    assert bits(permanent)==r['bits'][3] and bits(math.ldexp(permanent,-50))==r['bits'][4]
assert dict(Counter(str(r['status']) for r in a['rows']))==a['status_counts']=={'0':81,'1':5,'2':3}
assert len(a['environment_controls'])==4
for r in a['environment_controls']:
    assert r['rejected_status']==3 and r['restored_round']==0
    assert r['restored_mxcsr']&0xffc0==0x1f80
    if r['case']<4:assert r['changed_round'] in (1024,2048,3072)
    else:assert r['changed_mxcsr']==r['restored_mxcsr']|0x8040
out=dict(complete=True,fast_bit_matches=81,explicit_fallbacks=5,nonfinite_unknowns=3,environment_rejections=4,
         source_sha256=b['source_sha256'],library_sha256=b['library_sha256'],results_sha256=sha(C/'results.json'),
         new_compiled_calls=0,new_model_mass_or_random_calls=0,
         scope='Saved finite parity/enclosure and guarded environment evidence only; fallback and unknown remain explicit. No relative-point, speed, complete-cost or downstream mass guarantee.')
(HERE/'compiled_cofactor_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
