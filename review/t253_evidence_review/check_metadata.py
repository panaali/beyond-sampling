"""Independent JSON structure / Git metadata review; no score contractions."""
import hashlib
import json
from pathlib import Path
import subprocess

B = Path('/workspace/work/recover-small_h4_4/review/neuron_account_evidence')
P = Path(__file__).resolve().parent
M = json.loads((B/'manifest.json').read_text())
I = json.loads((B/'inventory.json').read_text())
rev = M['author_revision']
tree = subprocess.check_output(['git','ls-tree','--full-tree','-rl',rev,'--','m16/approaches/neuron_account/'],text=True)
actual = {}
for line in tree.splitlines():
    info, path = line.split('\t')
    mode, kind, blob, size = info.split()
    assert kind == 'blob'
    actual[path] = (blob, int(size))
assert set(actual) == {r['path'] for r in M['files']}
pins = {}
for r in M['files']:
    assert actual[r['path']] == (r['blob'],r['size'])
    if r['copied']:
        p = B/r['snapshot']; raw = p.read_bytes()
        assert len(raw) == r['size'] and hashlib.sha256(raw).hexdigest() == r['sha256']
        assert raw == subprocess.check_output(['git','show',rev+':'+r['path']])
        pins[str(p)] = r['sha256']
    else:
        assert r['path'].endswith('.npz') and 'snapshot' not in r
assert len(actual)==12 and len(pins)==10
for name in ['README.md','nacommon.py','phaseA.py','phaseA_modes.py','reportA.py']:
    assert (B/'snapshot'/name).read_bytes() == (P.parent/'neuron_intervention_source/snapshot'/name).read_bytes()
A = json.loads((B/'snapshot/results/phaseA.json').read_text())
O = json.loads((B/'snapshot/results/phaseA_modes.json').read_text())
metrics = {'acc','agree','acc_x9max','acc_x9second','acc_x9below','acc_blind'}
assert set(A)=={'N','seed_eval','seed_pilot','base','case_shares','matrices','groups'}
assert set(A['base'])==metrics and all(type(x) is float for x in A['base'].values())
assert set(A['case_shares'])=={'x9max','x9second','x9below','blind'}
assert all(type(x) is float for x in A['case_shares'].values())
arms={'zero','mean','resample'}
assert set(A['matrices'])==arms
for d in A['matrices'].values():
    assert set(d)==metrics
    for matrix in d.values():
        assert len(matrix)==16 and all(len(row)==10 for row in matrix)
        assert all(type(v) is float for row in matrix for v in row)
assert len(A['groups'])==22 and list(A['groups'])==I['phaseA']['group_names']
for d in A['groups'].values():
    assert set(d)==arms | {x+'_final_only' for x in arms}
    for entry in d.values():
        assert set(entry)==metrics and all(type(v) is float for v in entry.values())
assert set(O)=={'N','patterns'} and len(O['patterns'])==5
assert A['N']==I['phaseA']['declared_N']==200000
assert O['N']==I['modes']['declared_N']==400000
assert A['seed_eval']==I['phaseA']['declared_seed_eval']
assert A['seed_pilot']==I['phaseA']['declared_seed_pilot']
for row, expected in zip(O['patterns'],I['modes']['patterns']):
    assert set(row)=={'pattern','share','active','acc_in_pattern','modes'}
    assert row['pattern']==expected['pattern'] and len(row['active'])==expected['active_count']
    assert len(row['modes'])==expected['modes']
    assert all(type(v) is int for v in row['active'])
    assert type(row['share']) is float and type(row['acc_in_pattern']) is float
    for d in row['modes']:
        assert set(d)=={'abs','angle_deg','period','top_neurons','resample_h10','resample_h9','energy_share'}
        assert all(type(d[k]) is float for k in ['abs','angle_deg','energy_share'])
        assert d['period'] is None or type(d['period']) is float
        assert all(type(x) is str for x in d['top_neurons'])
        for k in ['resample_h10','resample_h9']:
            assert set(d[k])=={'acc','agree'} and all(type(v) is float for v in d[k].values())
assert I['complete'] and I['tree_files']==12 and I['copied_text_json_files']==10
assert I['prior_five_texts_identical']
for name in ['manifest.json','inventory.py','inventory.json','FINDINGS.md','README.md']:
    pins[str(B/name)]=hashlib.sha256((B/name).read_bytes()).hexdigest()
for name in ['README.md','check_metadata.py']:
    pins[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
print(json.dumps(dict(task='T253',reviewer='recover_34',complete=True,pass_all=True,
    execution_allowed=False,execution_authorized=False,author_package='dd4de999',
    owner_revision=rev,pins=pins,tree_entries=12,text_json_blobs=10,unopened_binary_entries=2,
    prior_source_texts_equal=5,score_contractions=0,author_calls=0,binary_payload_reads=0,
    scope='Written source/schema/provenance only; no score, runtime or checkpoint verification'),indent=2))
