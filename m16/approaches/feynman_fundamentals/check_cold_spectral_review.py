"""Independent saved-record audit; no spectral or model evaluation."""
import hashlib
import json
from pathlib import Path
import resource
from collections import Counter

resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
HERE = Path(__file__).resolve().parent
PEER = Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native')
S = PEER / 'spectral'
def read(p):
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

a = read(S/'capture.json')
original = read(PEER/'capture.json')
manifest = read(S/'capture_manifest.json')
assert a['complete'] and original['complete'] and len(a['rows']) == 12
assert a['manifest_sha256'] == sha(S/'capture_manifest.json')
assert a['runner_sha256'] == sha(S/'capture.py')
assert a['spy_binary_sha256'] == sha(S/'spy.so')
for p, h in manifest['files'].items():
    assert sha(Path(p)) == h, p
indices = [i for i,r in enumerate(original['rows']) if r['name'] in ('svd','leggauss')]
assert [r['saved_row'] for r in a['rows']] == indices
counts = Counter()
orders = []
zero_reflectors = 0
for r in a['rows']:
    old = original['rows'][r['saved_row']]
    assert old['kwargs'] == {'dict': []}
    assert r['name'] == old['name']
    assert r['parity'] and r['output_hex'] == [v['hex'] for v in old['output']['items']]
    events = r['events']
    assert 0 < len(events) <= 512
    counts.update(e['function'] for e in events)
    by = lambda name: [e['args'] for e in events if e['function'] == name]
    if r['name'] == 'svd':
        assert old['args']['items'][0]['shape'] == [1,4]
        assert by('dgesdd') == [[1,4,1,65], [1,4,0,65]]
        assert [1,4,0,0] in by('dgelqf')
        assert [4,4,1,0] in by('dorglq')
        assert by('dorgl2') == [[4,4,1,0]]
        assert by('dlarfg') == [[4,1,0,0], [1,1,1,0]]
    else:
        n = old['args']['items'][0]
        orders.append(n)
        assert by('dsyevd') == [[n,78,1,76], [n,78,0,76]]
        assert by('dsterf') == [[n,0,0,0]]
        assert by('dsytrd') == [[n,76,0,0]]
        assert by('dsytd2') == [[n,76,0,0]]
        assert by('dlarfg') == [[k,1,1,0] for k in range(n-1,0,-1)]
        zero_reflectors += n-1
assert orders == [6,3,3,6,10,16] and zero_reflectors == 38
summary = read(S/'summary.json')
assert summary['all12_parity'] and summary['SVD_calls'] == summary['rule_calls'] == 6
assert summary['total_legendre_reflector_calls'] == zero_reflectors
assert summary['legendre_eigen_orders'] == orders
source_upper = sum(559*n*(n-1)+368*n+6 for n in orders)
assert source_upper == 240946
result = dict(complete=True, records=12, all_saved_outputs_exact=True,
              query_inclusive_events=dict(counts), legendre_orders=orders,
              zero_legendre_reflectors=zero_reflectors,
              conditional_DSTERF_source_exclusive_upper=source_upper,
              hashes={str(p):sha(p) for p in (S/'capture.json', S/'spy.c', S/'DSTERF_SCOPE.md')},
              scope='Saved finite dispatch and parity only. DSTERF source bound excludes child prices and installed loop equivalence. Public DORGL2 version mismatch is preserved; no full spectral/native cost claim.',
              new_spectral_model_mass_calls=0)
(HERE/'cold_spectral_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
