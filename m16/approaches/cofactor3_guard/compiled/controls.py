"""One frozen25+64 parity run plus four restored environment controls."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/'results.json'
assert not OUT.exists(), 'Preserve all earlier attempts'
sys.path.insert(0,str(HERE.parent))
from primitive import enclose, exact_fallback, Ledger

def bits(x):
    return struct.pack('>d',x).hex()

def loaded_pins():
    pins = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and ('/libm.so' in path or '/libc.so' in path):
            pins[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return pins

t0 = time.process_time()
pins = json.loads((HERE/'control_pins.json').read_text())
for name,pin in pins.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin,name
build = json.loads((HERE/'build1/build_manifest.json').read_text())
assert hashlib.sha256((HERE/'build1/guard.so').read_bytes()).hexdigest()==build['library_sha256']
lib=C.CDLL(str(HERE/'build1/guard.so'))
lib.guard3.argtypes=[C.POINTER(C.c_double),C.POINTER(C.c_double)];lib.guard3.restype=C.c_int
lib.guard_getcsr.argtypes=[];lib.guard_getcsr.restype=C.c_uint
lib.guard_test_setcsr.argtypes=[C.c_uint];lib.guard_test_setcsr.restype=None
lib.guard_getround.argtypes=[];lib.guard_getround.restype=C.c_int
lib.guard_test_round.argtypes=[C.c_int];lib.guard_test_round.restype=C.c_int
round0,csr0=lib.guard_getround(),lib.guard_getcsr()
assert round0==0 and csr0&0xffc0==0x1f80
arr9,arr5=C.c_double*9,C.c_double*5
sentinel=bytes.fromhex('efbeaddebebafeca')*5

def call(matrix):
    a=arr9(*(x for row in matrix for x in row))
    o=arr5.from_buffer_copy(sentinel)
    status=lib.guard3(a,o)
    assert lib.guard_getround()==round0 and lib.guard_getcsr()&0xffc0==csr0&0xffc0
    if status: assert bytes(o)==sentinel
    return status,list(o)

rows=[]
parent=json.loads((HERE/'../results.json').read_text())
for saved in parent['rows']:
    a=[[float.fromhex(x) for x in row] for row in saved['input_hex']]
    status,got=call(a)
    prior=saved['result']; expected={'fast_enclosure':0,'exact_fallback':1,'unknown_nonfinite':2}[prior['status']]
    assert status==expected
    row={'name':saved['name'],'input_hex':saved['input_hex'],'status':status,'outputs_untouched':bool(status)}
    if status==0:
        keys=['point','lower','upper','permanent','error_allowance']
        want=[float.fromhex(prior[k]) for k in keys]
        assert [bits(x) for x in got]==[bits(x) for x in want],saved['name']
        row['bits']=[bits(x) for x in got]
    elif status==1:
        ledger=Ledger()
        fb=exact_fallback([x for r in a for x in r],ledger)
        keys=['point','lower','upper']
        assert [bits(fb[k]) for k in keys]==[bits(float.fromhex(prior[k])) for k in keys]
        assert fb['exact']==prior['exact']
        row['fallback_bits']=[bits(fb[k]) for k in keys]
        row['fallback_counts']=dict(ledger.counts)
        row['fallback_max_bits']=[ledger.max_numerator_bits,ledger.max_denominator_bits]
    rows.append(row)

saved_rows=json.loads((HERE/'../saved_minors/results.json').read_text())['rows']
for idx,saved in enumerate(saved_rows):
    a=[[float.fromhex(x) for x in row] for row in saved['minor_hex']]
    status,got=call(a);assert status==0
    old=enclose(a)
    want=[old[k] for k in ['point','lower','upper','permanent','error_allowance']]
    assert [bits(x) for x in got]==[bits(x) for x in want]
    point,lo,hi=got[:3]
    if saved['cofactor_sign']<0: point,lo,hi=-point,-hi,-lo
    old_signed=[saved['candidate_signed_point'],*saved['interval']]
    assert [bits(x) for x in [point,lo,hi]]==[bits(float.fromhex(x)) for x in old_signed]
    rows.append({'name':'saved_'+str(idx),'minor_hex':saved['minor_hex'],'status':status,
                 'bits':[bits(x) for x in got], 'signed_bits':[bits(x) for x in [point,lo,hi]],
                 'cofactor_sign':saved['cofactor_sign']})

# Allocate all floating objects before mutating modes. Only C/integer work occurs
# until finally restores both the fenv rounding and complete original MXCSR.
identity=arr9(1,0,0,0,1,0,0,0,1)
env=[]
for which in [1,2,3,4]:
    output=arr5.from_buffer_copy(sentinel)
    oldcsr,oldround=lib.guard_getcsr(),lib.guard_getround()
    assert oldround==0
    try:
        if which<4:
            rc=lib.guard_test_round(which)
            assert rc==0
        else:
            lib.guard_test_setcsr(oldcsr|0x8040)
        changedcsr,changedround=lib.guard_getcsr(),lib.guard_getround()
        status=lib.guard3(identity,output)
    finally:
        rc=lib.guard_test_round(0)
        lib.guard_test_setcsr(oldcsr)
        assert rc==0 and lib.guard_getround()==oldround and lib.guard_getcsr()==oldcsr
    assert status==3 and bytes(output)==sentinel
    env.append({'case':which,'rejected_status':status,'changed_round':changedround,
                'changed_mxcsr':changedcsr,'restored_round':oldround,'restored_mxcsr':oldcsr})
assert len(rows)==89
summary={str(s):sum(r['status']==s for r in rows) for s in [0,1,2]}
assert summary=={'0':81,'1':5,'2':3}
result={'complete':True,'rows':rows,'environment_controls':env,'status_counts':summary,
        'build':build,'source_pins':pins,'runtime_libc_libm':loaded_pins(),
        'initial_round':round0,'initial_mxcsr':csr0,'cpu_seconds':time.process_time()-t0,
        'kernel_calls':93,'python_fallback_calls':5,'python_fast_comparator_calls':64,
        'new_native_determinant_mass_model_MC_calls':0,
        'scope':'Existing finite parity and guarded runtime modes only; no full cost/speed or downstream mass claim'}
with OUT.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['complete','status_counts','kernel_calls','cpu_seconds']},indent=2))
