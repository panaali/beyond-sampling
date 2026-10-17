"""Independent saved layout/copy and permutation-determinant audit; no kernels."""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import hashlib, json, struct, time

P = Path('/workspace/work/recover-small_h2_78/m16/approaches/orth5_cost_review/determinant_layout')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n): return json.loads((P/n).read_text())
def det(a):
    n = len(a)
    total = F(0)
    for perm in permutations(range(n)):
        term = F((-1)**sum(perm[i] > perm[j] for i in range(n) for j in range(i+1,n)))
        for i,j in enumerate(perm): term *= F(a[i][j])
        total += term
    return total

start = time.process_time()
b, o = read('baseline.json'), read('observed.json')
assert b['complete'] and o['complete'] and len(b['rows']) == len(o['rows']) == 12
assert b['mode'] == 'baseline' and o['mode'] == 'observed'
assert o['baseline_sha256'] == sha(P/'baseline.json')
assert b['source_pins'] == o['source_pins']
for n,h in b['source_pins'].items(): assert sha(P/n) == h
for n,h in read('pins.json').items(): assert sha(P/n) == h
build = read('build.json')
assert build['complete'] and build['returncode'] == 0
for n,h in build['source_sha256'].items(): assert sha(P/n) == h
assert sha(P/'spy.so') == build['library_sha256']
assert sha(P/'spy_disassembly.txt') == build['disassembly_sha256']
assert b['runtime'] == o['runtime']
rt = b['runtime']; env = read('det_environment.json'); dispatch = read('runtime_dispatch.json')
assert rt['threads'] == 1 and rt['unique_zero_offset_mapping']
assert rt['binary_hashes'] == env['binary_hashes']
for n,h in env['binary_hashes'].items(): assert sha(Path(env['binary_paths'][n])) == h
assert rt['blocks'] == dispatch['block_parameter_offsets']
assert rt['functions'] == [dict(table_offset=int(e['table_offset'],16),target_relative=int(e['target_relative_elf_offset'],16)) for e in dispatch['functions']]

bases = {3:[[4.,1.,2.],[1.,5.,1.],[2.,1.,6.]],4:[[4.,1.,2.,1.],[1.,5.,1.,2.],[2.,1.,6.,1.],[1.,2.,1.,7.]]}
index = 0; matrices = singular = success = 0; summaries = []
for d, dense in bases.items():
    patterns = [('identity',[[float(i == j) for j in range(d)] for i in range(d)]),
                ('permuted_dense',dense[-1:] + dense[:-1]),
                ('zero_row',dense[:-1] + [[0.]*d])]
    for pattern, base in patterns:
        for batch in (False, True):
            br, row = b['rows'][index], o['rows'][index]; index += 1
            count = 128 if batch else 1
            name = f'd{d}_{pattern}_' + ('batch' if batch else 'scalar')
            expected = []
            for j in range(count):
                a = [r[:] for r in base]
                if batch: a[0][0] += j/128
                expected.append(a)
            hexes = [[[x.hex() for x in r] for r in a] for a in expected]
            expected_hash = hashlib.sha256(b''.join(struct.pack('d',x) for a in expected for r in a for x in r)).hexdigest()
            for r in (br,row):
                assert r['name'] == name and r['pattern'] == pattern
                assert r['shape'] == ([128,d,d] if batch else [d,d])
                assert r['strides'] == ([8,1024,1024*d] if batch else [8*d,8])
                assert r['dtype'] == 'float64' and r['matrix_hex'] == hexes
                assert r['input_sha256'] == expected_hash
            for key in ('output_hex','exact_determinants','reference_checks'): assert br[key] == row[key]
            assert len(row['events']) == len(row['output_hex']) == len(row['exact_determinants']) == count
            for j,(a,event) in enumerate(zip(expected,row['events'])):
                exact = det(a)
                assert exact == F(row['exact_determinants'][j])
                val = float.fromhex(row['output_hex'][j])
                check = val == 0 if exact == 0 else abs(F(val)-exact) <= F(1,10**12)*abs(exact)
                assert check and row['reference_checks'][j] is True
                assert event['matrix_hex'] == hexes[j]
                assert (event['m'],event['n'],event['lda']) == (d,d,d)
                assert len(event['pivots']) == d and all(type(x) is int and 1 <= x <= d for x in event['pivots'])
                if exact == 0:
                    assert pattern == 'zero_row' and 1 <= event['info'] <= d
                    singular += 1
                else:
                    assert event['info'] == 0
                    success += 1
                matrices += 1
            summaries.append(dict(name=name,matrices=count,copy_and_reference_pass=True))
assert matrices == 774 and singular == 258 and success == 516
assert b['mass_model_MC_calls'] == o['mass_model_MC_calls'] == 0
out = dict(complete=True,arrays=12,observed_events=matrices,successful=success,singular=singular,
           baseline_sha256=sha(P/'baseline.json'),observed_sha256=sha(P/'observed.json'),
           source_pins=b['source_pins'],rows=summaries,review_cpu_seconds=time.process_time()-start,
           new_native_or_model_calls=0,
           scope='All saved binary matrices independently reconstructed, determinant references checked by permutation expansion, logical copies/order, dimensions, pivots validity, status and output parity checked. Runtime/source pins support these fixed layouts only; no complete native cost or general corpus guarantee.')
Path(__file__).with_name('orth5_determinant_layout_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
